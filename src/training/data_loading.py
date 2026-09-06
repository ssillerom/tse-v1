"""Build deterministic train and validation inputs for one finite experiment."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
from torch.utils.data import DataLoader, Subset

from src.data.dataset import PretrainingDataset
from src.data.manifest import DataManifest, load_manifest
from src.data.mixture import DeterministicMixtureSampler, MixtureDataset
from src.data.recipe import TrainingRecipe

from .checkpoint import RestoredCheckpoint
from .evaluation import evenly_spaced_validation_indices
from .trainer import Batch


@dataclass(frozen=True)
class DatasetBundle:
    """Training data plus the fixed held-out distributions used for evaluation."""

    train_dataset: PretrainingDataset | MixtureDataset
    validation_batches: Iterable[Batch] | Mapping[str, Iterable[Batch]]
    validation_weights: Mapping[str, float] | None


@dataclass
class TrainingInputs:
    """Reiterable batches and exact counters needed to continue a run."""

    train_loader: Iterable[Batch]
    sequences_per_step: int
    batches_start_step: int
    tokens_seen_at_start: int
    sampler: DeterministicMixtureSampler | None = None
    source_tokens_seen: dict[str, int] | None = None


def build_validation_loader(
    manifest_path: Path | DataManifest,
    *,
    seq_len: int,
    batch_size: int,
    eval_batches: int | None,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    """Build a fixed-cost loader whose samples span the full validation split."""
    dataset = PretrainingDataset(
        manifest_path=manifest_path,
        split="validation",
        seq_len=seq_len,
    )
    requested_samples = len(dataset) if eval_batches is None else eval_batches * batch_size
    indices = evenly_spaced_validation_indices(
        dataset_size=len(dataset),
        sample_size=requested_samples,
    )
    # Subset freezes the exact sample identities. Recreating this loader at a
    # later checkpoint therefore evaluates the same windows in the same order.
    validation_subset = Subset(dataset, indices)
    return DataLoader(
        validation_subset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def build_dataset_bundle(
    *,
    manifest_path: Path | DataManifest | None,
    recipe: TrainingRecipe | None,
    seq_len: int,
    batch_size: int,
    eval_batches: int | None,
    num_workers: int,
    pin_memory: bool,
) -> DatasetBundle:
    """Open train sources and matching deterministic validation subsets."""
    if recipe is None:
        if manifest_path is None:
            raise RuntimeError("a manifest path is required without a recipe")
        manifest = (
            manifest_path
            if isinstance(manifest_path, DataManifest)
            else load_manifest(manifest_path)
        )
        return DatasetBundle(
            train_dataset=PretrainingDataset(
                manifest_path=manifest,
                split="train",
                seq_len=seq_len,
            ),
            validation_batches=build_validation_loader(
                manifest,
                seq_len=seq_len,
                batch_size=batch_size,
                eval_batches=eval_batches,
                num_workers=num_workers,
                pin_memory=pin_memory,
            ),
            validation_weights=None,
        )

    return DatasetBundle(
        train_dataset=MixtureDataset(
            {
                source.name: PretrainingDataset(
                    manifest_path=source.manifest,
                    split="train",
                    seq_len=seq_len,
                )
                for source in recipe.sources
            }
        ),
        validation_batches={
            source.name: build_validation_loader(
                source.manifest,
                seq_len=seq_len,
                batch_size=batch_size,
                eval_batches=eval_batches,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
            for source in recipe.sources
        },
        validation_weights=recipe.source_weights,
    )


def build_training_inputs(
    *,
    bundle: DatasetBundle,
    recipe: TrainingRecipe | None,
    restored_checkpoint: RestoredCheckpoint | None,
    start_step: int,
    seq_len: int,
    batch_size: int,
    grad_accum_steps: int,
    seed: int,
    num_workers: int,
    pin_memory: bool,
) -> TrainingInputs:
    """Position deterministic training data at the first unfinished sequence."""
    sequences_per_step = batch_size * grad_accum_steps
    if recipe is None:
        return TrainingInputs(
            train_loader=DataLoader(
                bundle.train_dataset,
                batch_size=batch_size,
                shuffle=False,
                drop_last=True,
                num_workers=num_workers,
                pin_memory=pin_memory,
            ),
            sequences_per_step=sequences_per_step,
            batches_start_step=0,
            tokens_seen_at_start=0,
        )

    expected_data_position = start_step * sequences_per_step
    restored_data_position = 0 if restored_checkpoint is None else restored_checkpoint.data_position
    if restored_data_position != expected_data_position:
        raise ValueError(
            "Recipe checkpoint data_position does not match its completed steps: "
            f"expected {expected_data_position}, got {restored_data_position}"
        )
    mixture_dataset = cast(MixtureDataset, bundle.train_dataset)
    sampler = DeterministicMixtureSampler(
        dataset=mixture_dataset,
        phases=recipe.phases,
        seq_len=seq_len,
        seed=seed,
        start_position=restored_data_position,
    )
    tokens_seen_at_start = 0 if restored_checkpoint is None else restored_checkpoint.tokens_seen
    restored_source_tokens = (
        None if restored_checkpoint is None else restored_checkpoint.source_tokens_seen
    )
    if restored_source_tokens is not None:
        if set(restored_source_tokens) != {source.name for source in recipe.sources}:
            raise ValueError("checkpoint source_tokens_seen does not match recipe sources")
        source_tokens_seen = dict(restored_source_tokens)
    else:
        restored_source_sequences = sampler.source_sequence_counts(0, restored_data_position)
        source_tokens_seen = {
            source_name: sequence_count * seq_len
            for source_name, sequence_count in restored_source_sequences.items()
        }
    if sum(source_tokens_seen.values()) != tokens_seen_at_start:
        raise ValueError("recipe source token counts do not match checkpoint tokens_seen")

    return TrainingInputs(
        train_loader=DataLoader(
            mixture_dataset,
            batch_size=batch_size,
            sampler=sampler,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        sequences_per_step=sequences_per_step,
        batches_start_step=start_step,
        tokens_seen_at_start=tokens_seen_at_start,
        sampler=sampler,
        source_tokens_seen=source_tokens_seen,
    )
