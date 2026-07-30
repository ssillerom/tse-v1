"""Train and evaluate the V1 decoder-only language model."""

import argparse
import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import asdict, replace
from pathlib import Path
from typing import Literal, cast

import tiktoken
import torch
import wandb
from torch.utils.data import DataLoader

from src.data.dataset import PretrainingDataset
from src.data.manifest import ManifestTokenizer, load_manifest
from src.data.mixture import DeterministicMixtureSampler, MixtureDataset
from src.data.recipe import TrainingRecipe, load_training_recipe
from src.model.config import ModelConfig
from src.model.gpt import GPT
from src.training.checkpoint import (
    TrainingRunConfig,
    restore_checkpoint,
    restore_latest_checkpoint,
    save_checkpoint,
)
from src.training.evaluation import perplexity_from_loss
from src.training.optimizer import build_adamw_parameter_groups
from src.training.rng import capture_torch_rng_state, restore_torch_rng_state
from src.training.trainer import (
    Batch,
    LearningRateSchedule,
    Precision,
    StepMetrics,
    TrainingConfig,
    evaluate,
    evaluate_domains,
    train,
)
from src.training.wandb_logging import (
    WandbEvaluationLogger,
    WandbRun,
    domain_evaluation_to_wandb,
)

WandbMode = Literal["online", "offline", "disabled"]
PrecisionArgument = Literal["auto", "fp32", "bf16"]
ADAMW_BETAS = (0.9, 0.95)
ADAMW_EPS = 1e-8
DEFAULT_WANDB_PROJECT = "llm-from-scratch"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    data_source = parser.add_mutually_exclusive_group(required=True)
    data_source.add_argument("--manifest", type=Path)
    data_source.add_argument("--recipe", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/v1"))
    parser.add_argument(
        "--resume",
        help="checkpoint path to restore, or 'latest' for the newest valid checkpoint",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument(
        "--precision",
        choices=("auto", "fp32", "bf16"),
        default="auto",
        help="auto selects bf16 on CUDA and fp32 elsewhere",
    )
    parser.add_argument(
        "--compile",
        dest="compile_model",
        action="store_true",
        help="compile the model forward/backward graph on CUDA",
    )
    parser.add_argument(
        "--compile-mode",
        choices=("default", "reduce-overhead", "max-autotune"),
        help="TorchInductor mode; requires --compile and defaults to 'default'",
    )
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--vocab-size", type=int, default=50_304)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.0)

    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--max-steps",
        type=int,
        help="optimizer-step limit; defaults to 500 for one manifest or the recipe budget",
    )
    parser.add_argument(
        "--stop-after-step",
        type=int,
        help="finish this process early while keeping the full schedule resumable",
    )
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--warmup-steps", type=int, default=25)
    parser.add_argument("--max-learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--min-learning-rate",
        type=float,
        help="defaults to 3e-5 for cosine and 0 for WSD",
    )
    parser.add_argument(
        "--lr-schedule",
        choices=("cosine", "wsd"),
        default="cosine",
    )
    parser.add_argument(
        "--decay-start-step",
        type=int,
        help="first WSD decay step; inferred from the final recipe phase when omitted",
    )
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")

    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--sample-interval", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--keep-last-checkpoints", type=int, default=3)

    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-name")
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="online",
    )
    return parser


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is not available")
    return torch.device(requested)


def _resolve_precision(requested: PrecisionArgument, device: torch.device) -> Precision:
    if requested == "auto":
        return "bf16" if device.type == "cuda" and torch.cuda.is_bf16_supported() else "fp32"
    if requested == "bf16" and device.type == "mps":
        raise ValueError("bf16 precision is not supported by this trainer on MPS")
    if requested == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise ValueError("bf16 precision was requested but is not supported by this CUDA device")
    return requested


def _load_encoding_from_tokenizer(tokenizer: ManifestTokenizer) -> tiktoken.Encoding:
    encoding = tiktoken.get_encoding(tokenizer.encoding_name)
    if encoding.eot_token != tokenizer.eot_token_id:
        raise ValueError(
            f"Tokenizer eot_token={tokenizer.eot_token_id} does not match "
            f"encoding {tokenizer.encoding_name!r}"
        )
    return encoding


def _load_encoding(manifest_path: Path) -> tiktoken.Encoding:
    tokenizer = load_manifest(manifest_path).tokenizer
    if tokenizer is None:
        raise ValueError("Manifest must declare tokenizer metadata for evaluation")
    return _load_encoding_from_tokenizer(tokenizer)


def _manifest_sha256(manifest_path: Path) -> str:
    with manifest_path.open("rb") as manifest_file:
        return hashlib.file_digest(manifest_file, "sha256").hexdigest()


def _recipe_sha256(recipe: TrainingRecipe) -> str:
    """Fingerprint the recipe contract and every referenced manifest."""
    digest = hashlib.sha256()
    digest.update(recipe.path.read_bytes())
    for source in recipe.sources:
        digest.update(b"\0")
        digest.update(source.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(source.manifest_path.read_bytes())
    return digest.hexdigest()


def _resolve_recipe_steps(
    recipe: TrainingRecipe,
    seq_len: int,
    batch_size: int,
    grad_accum_steps: int,
    requested_max_steps: int | None,
) -> int:
    tokens_per_step = seq_len * batch_size * grad_accum_steps
    available_steps = recipe.total_tokens // tokens_per_step
    if available_steps == 0:
        raise ValueError(f"Recipe {recipe.name!r} does not contain one complete optimizer step")
    if requested_max_steps is None:
        return available_steps
    if requested_max_steps > available_steps:
        raise ValueError(
            f"max_steps={requested_max_steps} exceeds the {available_steps} complete "
            f"optimizer steps in recipe {recipe.name!r}"
        )
    return requested_max_steps


def _resolve_decay_start_step(
    schedule: LearningRateSchedule,
    requested_decay_start_step: int | None,
    recipe: TrainingRecipe | None,
    max_steps: int,
    seq_len: int,
    batch_size: int,
    grad_accum_steps: int,
) -> int | None:
    if schedule == "cosine":
        return requested_decay_start_step
    if requested_decay_start_step is not None:
        return requested_decay_start_step
    if recipe is None or len(recipe.phases) == 1:
        return max_steps
    stable_tokens = sum(phase.target_tokens for phase in recipe.phases[:-1])
    tokens_per_step = seq_len * batch_size * grad_accum_steps
    return min(stable_tokens // tokens_per_step, max_steps)


def main(argv: list[str] | None = None) -> int:
    """Run one finite V1 experiment."""
    arguments = _build_parser().parse_args(argv)
    if arguments.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if arguments.grad_accum_steps <= 0:
        raise ValueError("grad_accum_steps must be positive")
    if arguments.num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    if arguments.checkpoint_interval <= 0:
        raise ValueError("checkpoint_interval must be positive")
    if arguments.keep_last_checkpoints <= 0:
        raise ValueError("keep_last_checkpoints must be positive")

    device = _resolve_device(arguments.device)
    precision = _resolve_precision(cast(PrecisionArgument, arguments.precision), device)
    if arguments.compile_model and device.type != "cuda":
        raise ValueError("torch.compile training is enabled only on CUDA")
    if arguments.compile_mode is not None and not arguments.compile_model:
        raise ValueError("compile_mode requires --compile")
    compile_mode = (
        None
        if not arguments.compile_model
        else arguments.compile_mode
        if arguments.compile_mode is not None
        else "default"
    )
    torch.manual_seed(arguments.seed)
    if device.type == "mps":
        torch.mps.manual_seed(arguments.seed)

    recipe = None if arguments.recipe is None else load_training_recipe(arguments.recipe)
    encoding = (
        _load_encoding(arguments.manifest)
        if recipe is None
        else _load_encoding_from_tokenizer(recipe.tokenizer)
    )
    if arguments.vocab_size < encoding.n_vocab:
        raise ValueError(
            f"vocab_size={arguments.vocab_size} is smaller than tokenizer vocabulary "
            f"{encoding.n_vocab}"
        )

    model_config = ModelConfig(
        vocab_size=arguments.vocab_size,
        d_model=arguments.d_model,
        n_layers=arguments.n_layers,
        n_heads=arguments.n_heads,
        max_seq_len=arguments.seq_len,
        dropout=arguments.dropout,
        use_sdpa=True,
    )
    max_steps = (
        arguments.max_steps
        if arguments.max_steps is not None
        else 500
        if recipe is None
        else _resolve_recipe_steps(
            recipe=recipe,
            seq_len=model_config.max_seq_len,
            batch_size=arguments.batch_size,
            grad_accum_steps=arguments.grad_accum_steps,
            requested_max_steps=None,
        )
    )
    if recipe is not None and arguments.max_steps is not None:
        max_steps = _resolve_recipe_steps(
            recipe=recipe,
            seq_len=model_config.max_seq_len,
            batch_size=arguments.batch_size,
            grad_accum_steps=arguments.grad_accum_steps,
            requested_max_steps=arguments.max_steps,
        )
    schedule = cast(LearningRateSchedule, arguments.lr_schedule)
    min_learning_rate = (
        arguments.min_learning_rate
        if arguments.min_learning_rate is not None
        else 0.0
        if schedule == "wsd"
        else 3e-5
    )
    decay_start_step = _resolve_decay_start_step(
        schedule=schedule,
        requested_decay_start_step=arguments.decay_start_step,
        recipe=recipe,
        max_steps=max_steps,
        seq_len=model_config.max_seq_len,
        batch_size=arguments.batch_size,
        grad_accum_steps=arguments.grad_accum_steps,
    )
    training_config = TrainingConfig(
        max_steps=max_steps,
        grad_accum_steps=arguments.grad_accum_steps,
        warmup_steps=arguments.warmup_steps,
        max_learning_rate=arguments.max_learning_rate,
        min_learning_rate=min_learning_rate,
        max_grad_norm=arguments.max_grad_norm,
        eval_interval=arguments.eval_interval,
        eval_batches=arguments.eval_batches,
        precision=precision,
        learning_rate_schedule=schedule,
        decay_start_step=decay_start_step,
    )
    end_step = (
        training_config.max_steps
        if arguments.stop_after_step is None
        else arguments.stop_after_step
    )
    if not 0 <= end_step <= training_config.max_steps:
        raise ValueError(
            "stop_after_step must be between 0 and "
            f"max_steps={training_config.max_steps}, got {end_step}"
        )
    data_contract_sha256 = (
        _manifest_sha256(arguments.manifest) if recipe is None else _recipe_sha256(recipe)
    )
    run_config = TrainingRunConfig(
        data_contract_sha256=data_contract_sha256,
        seed=arguments.seed,
        batch_size=arguments.batch_size,
        optimizer_name="AdamW",
        optimizer_betas=ADAMW_BETAS,
        optimizer_weight_decay=arguments.weight_decay,
        optimizer_eps=ADAMW_EPS,
        compile_mode=compile_mode,
        source_token_budgets=(
            None if recipe is None else tuple(recipe.source_token_totals.items())
        ),
    )
    if recipe is None:
        train_dataset: PretrainingDataset | MixtureDataset = PretrainingDataset(
            manifest_path=arguments.manifest,
            split="train",
            seq_len=model_config.max_seq_len,
        )
        validation_batches: Iterable[Batch] | Mapping[str, Iterable[Batch]] = DataLoader(
            PretrainingDataset(
                manifest_path=arguments.manifest,
                split="validation",
                seq_len=model_config.max_seq_len,
            ),
            batch_size=arguments.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=arguments.num_workers,
            pin_memory=arguments.pin_memory,
        )
        validation_weights = None
    else:
        train_dataset = MixtureDataset(
            {
                source.name: PretrainingDataset(
                    manifest_path=source.manifest_path,
                    split="train",
                    seq_len=model_config.max_seq_len,
                )
                for source in recipe.sources
            }
        )
        validation_batches = {
            source.name: DataLoader(
                PretrainingDataset(
                    manifest_path=source.manifest_path,
                    split="validation",
                    seq_len=model_config.max_seq_len,
                ),
                batch_size=arguments.batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=arguments.num_workers,
                pin_memory=arguments.pin_memory,
            )
            for source in recipe.sources
        }
        validation_weights = recipe.source_weights

    model = GPT(model_config).to(device)
    optimizer = torch.optim.AdamW(
        build_adamw_parameter_groups(model, weight_decay=arguments.weight_decay),
        lr=training_config.max_learning_rate,
        betas=run_config.optimizer_betas,
        eps=run_config.optimizer_eps,
    )
    restored_checkpoint = None
    if arguments.resume is not None:
        if arguments.resume == "latest":
            restored_checkpoint = restore_latest_checkpoint(
                directory=arguments.checkpoint_dir,
                model=model,
                optimizer=optimizer,
                training_config=training_config,
                map_location="cpu",
                run_config=run_config,
            )
        else:
            restored_checkpoint = restore_checkpoint(
                path=arguments.resume,
                model=model,
                optimizer=optimizer,
                training_config=training_config,
                map_location="cpu",
                run_config=run_config,
            )
    training_model = (
        model
        if compile_mode is None
        else cast(
            GPT,
            torch.compile(
                model,
                mode=compile_mode,
                dynamic=False,
            ),
        )
    )
    start_step = 0 if restored_checkpoint is None else restored_checkpoint.step
    sequences_per_step = arguments.batch_size * training_config.grad_accum_steps
    expected_data_position = start_step * sequences_per_step
    sampler: DeterministicMixtureSampler | None = None
    source_tokens_seen: dict[str, int] | None = None
    if recipe is not None:
        restored_data_position = (
            0 if restored_checkpoint is None else restored_checkpoint.data_position
        )
        if restored_data_position != expected_data_position:
            raise ValueError(
                "Recipe checkpoint data_position does not match its completed steps: "
                f"expected {expected_data_position}, got {restored_data_position}"
            )
        sampler = DeterministicMixtureSampler(
            dataset=cast(MixtureDataset, train_dataset),
            phases=recipe.phases,
            seq_len=model_config.max_seq_len,
            seed=arguments.seed,
            start_position=restored_data_position,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=arguments.batch_size,
            sampler=sampler,
            drop_last=True,
            num_workers=arguments.num_workers,
            pin_memory=arguments.pin_memory,
        )
        batches_start_step = start_step
        tokens_seen_at_start = 0 if restored_checkpoint is None else restored_checkpoint.tokens_seen
        restored_source_tokens = (
            None if restored_checkpoint is None else restored_checkpoint.source_tokens_seen
        )
        if restored_source_tokens is not None:
            if set(restored_source_tokens) != {source.name for source in recipe.sources}:
                raise ValueError("checkpoint source_tokens_seen does not match recipe sources")
            source_tokens_seen = dict(restored_source_tokens)
        else:
            restored_source_sequences = sampler.source_sequence_counts(
                0,
                restored_data_position,
            )
            source_tokens_seen = {
                source_name: sequence_count * model_config.max_seq_len
                for source_name, sequence_count in restored_source_sequences.items()
            }
        if sum(source_tokens_seen.values()) != tokens_seen_at_start:
            raise ValueError("recipe source token counts do not match checkpoint tokens_seen")
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=arguments.batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=arguments.num_workers,
            pin_memory=arguments.pin_memory,
        )
        batches_start_step = 0
        tokens_seen_at_start = 0
    resume_rng_snapshot = None if restored_checkpoint is None else capture_torch_rng_state()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    run_name = arguments.wandb_name or (
        f"v1-{model_config.n_layers}l-{model_config.d_model}d-{arguments.seed}"
    )
    wandb_mode = cast(WandbMode, arguments.wandb_mode)
    stored_wandb_project = (
        None if restored_checkpoint is None else restored_checkpoint.wandb_project
    )
    stored_wandb_entity = None if restored_checkpoint is None else restored_checkpoint.wandb_entity
    if (
        stored_wandb_project is not None
        and arguments.wandb_project is not None
        and arguments.wandb_project != stored_wandb_project
    ):
        raise ValueError("wandb_project does not match the project stored in the checkpoint")
    if (
        stored_wandb_entity is not None
        and arguments.wandb_entity is not None
        and arguments.wandb_entity != stored_wandb_entity
    ):
        raise ValueError("wandb_entity does not match the entity stored in the checkpoint")
    wandb_project = stored_wandb_project or arguments.wandb_project or DEFAULT_WANDB_PROJECT
    wandb_entity = stored_wandb_entity or arguments.wandb_entity
    resume_wandb_run = (
        restored_checkpoint is not None and restored_checkpoint.wandb_run_id is not None
    )

    with wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        name=run_name,
        mode=wandb_mode,
        id=None if restored_checkpoint is None else restored_checkpoint.wandb_run_id,
        resume="must" if resume_wandb_run else None,
        config={
            "model": asdict(model_config),
            "training": asdict(training_config),
            "optimizer": {
                "name": "AdamW",
                "betas": list(run_config.optimizer_betas),
                "eps": run_config.optimizer_eps,
                "weight_decay": run_config.optimizer_weight_decay,
            },
            "data": (
                {
                    "manifest": str(arguments.manifest),
                    "batch_size": arguments.batch_size,
                }
                if recipe is None
                else {
                    "recipe": str(recipe.path),
                    "recipe_name": recipe.name,
                    "recipe_tokens": recipe.total_tokens,
                    "validation_weights": recipe.source_weights,
                    "sources": {
                        source.name: str(source.manifest_path) for source in recipe.sources
                    },
                    "phases": [
                        {
                            "name": phase.name,
                            "tokens": phase.target_tokens,
                            "source_tokens": phase.source_token_map,
                        }
                        for phase in recipe.phases
                    ],
                    "batch_size": arguments.batch_size,
                }
            ),
            "seed": arguments.seed,
            "device": str(device),
            "parameter_count": parameter_count,
            "torch_compile": {
                "enabled": compile_mode is not None,
                "mode": compile_mode,
                "dynamic": False if compile_mode is not None else None,
            },
        },
    ) as raw_run:
        run = cast(WandbRun, raw_run)
        wandb_run_id = (
            restored_checkpoint.wandb_run_id
            if restored_checkpoint is not None and restored_checkpoint.wandb_run_id is not None
            else run.id
        )
        checkpoint_wandb_project = (
            wandb_project
            if wandb_mode == "disabled"
            else getattr(run, "project", None) or wandb_project
        )
        checkpoint_wandb_entity = (
            wandb_entity
            if wandb_mode == "disabled"
            else getattr(run, "entity", None) or wandb_entity
        )
        logger = WandbEvaluationLogger(
            run=run,
            model=model,
            encoding=encoding,
            device=device,
            sample_interval=arguments.sample_interval,
            max_new_tokens=arguments.max_new_tokens,
        )
        if isinstance(validation_batches, Mapping):
            if validation_weights is None:
                raise RuntimeError("recipe validation has no source weights")
            initial_validation = evaluate_domains(
                model=model,
                batches_by_domain=validation_batches,
                domain_weights=validation_weights,
                device=device,
                max_batches=training_config.eval_batches,
                precision=training_config.precision,
            )
            initial_validation_payload: dict[str, object] = dict(
                domain_evaluation_to_wandb(initial_validation, start_step)
            )
            if source_tokens_seen is not None:
                for source, token_count in source_tokens_seen.items():
                    initial_validation_payload[f"trainer/source_tokens_seen/{source}"] = token_count
            run.log(initial_validation_payload)
        else:
            initial_validation_loss = evaluate(
                model=model,
                batches=validation_batches,
                device=device,
                max_batches=training_config.eval_batches,
                precision=training_config.precision,
            )
            run.log(
                {
                    "trainer/global_step": start_step,
                    "validation/loss": initial_validation_loss,
                    "validation/perplexity": perplexity_from_loss(initial_validation_loss),
                }
            )
        logger.log_samples(step=start_step)

        arguments.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        def on_step(metrics: StepMetrics) -> None:
            logged_metrics = metrics
            if sampler is not None and source_tokens_seen is not None:
                step_end_position = metrics.step * sequences_per_step
                step_start_position = step_end_position - sequences_per_step
                step_source_sequences = sampler.source_sequence_counts(
                    step_start_position,
                    step_end_position,
                )
                for source, sequence_count in step_source_sequences.items():
                    source_tokens_seen[source] += sequence_count * model_config.max_seq_len
                logged_metrics = replace(
                    metrics,
                    source_tokens_seen=dict(source_tokens_seen),
                )
            if metrics.step % arguments.checkpoint_interval == 0:
                save_checkpoint(
                    path=arguments.checkpoint_dir / f"step_{metrics.step:06d}.pt",
                    model=model,
                    optimizer=optimizer,
                    step=metrics.step,
                    training_config=training_config,
                    wandb_run_id=wandb_run_id,
                    wandb_project=checkpoint_wandb_project,
                    wandb_entity=checkpoint_wandb_entity,
                    keep_last_n=arguments.keep_last_checkpoints,
                    run_config=run_config,
                    data_position=metrics.step * sequences_per_step,
                    tokens_seen=logged_metrics.tokens_seen,
                    source_tokens_seen=logged_metrics.source_tokens_seen,
                )
            logger(logged_metrics)

        if resume_rng_snapshot is not None:
            restore_torch_rng_state(resume_rng_snapshot)

        history = train(
            model=training_model,
            optimizer=optimizer,
            train_batches=train_loader,
            config=training_config,
            device=device,
            validation_batches=validation_batches,
            validation_weights=validation_weights,
            start_step=start_step,
            end_step=end_step,
            on_step=on_step,
            batches_start_step=batches_start_step,
            tokens_seen_at_start=tokens_seen_at_start,
        )
        final_step = history[-1].step if history else start_step
        if final_step % arguments.checkpoint_interval != 0:
            save_checkpoint(
                path=arguments.checkpoint_dir / f"step_{final_step:06d}.pt",
                model=model,
                optimizer=optimizer,
                step=final_step,
                training_config=training_config,
                wandb_run_id=wandb_run_id,
                wandb_project=checkpoint_wandb_project,
                wandb_entity=checkpoint_wandb_entity,
                keep_last_n=arguments.keep_last_checkpoints,
                run_config=run_config,
                data_position=final_step * sequences_per_step,
                tokens_seen=(history[-1].tokens_seen if history else tokens_seen_at_start),
                source_tokens_seen=source_tokens_seen,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
