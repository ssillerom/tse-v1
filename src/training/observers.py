"""Translate completed optimizer steps into durable experiment observations."""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

import torch

from src.data.mixture import DeterministicMixtureSampler
from src.model.gpt import GPT
from src.training.checkpoint import TrainingRunConfig, save_checkpoint
from src.training.trainer import StepMetrics, TrainingConfig
from src.training.wandb_logging import WandbEvaluationLogger


@dataclass
class CheckpointWriter:
    """Persist resumable steps and the best validated model through one interface."""

    directory: Path
    model: GPT
    optimizer: torch.optim.Optimizer
    training_config: TrainingConfig
    run_config: TrainingRunConfig
    sequences_per_step: int
    wandb_run_id: str | None
    wandb_project: str | None
    wandb_entity: str | None
    keep_last_n: int
    best_validation_loss: float | None = None

    def save_best_if_improved(
        self,
        *,
        step: int,
        tokens_seen: int,
        source_tokens_seen: Mapping[str, int] | None,
        validation_loss: float,
    ) -> bool:
        """Replace best_validation.pt only when the held-out loss improves."""
        if self.best_validation_loss is not None and validation_loss >= self.best_validation_loss:
            return False
        self._save(
            path=self.directory / "best_validation.pt",
            step=step,
            tokens_seen=tokens_seen,
            source_tokens_seen=source_tokens_seen,
            validation_loss=validation_loss,
            best_validation_loss=validation_loss,
            keep_last_n=None,
        )
        # Advance the in-memory threshold only after the atomic publish succeeds.
        self.best_validation_loss = validation_loss
        return True

    def save_step(
        self,
        *,
        step: int,
        tokens_seen: int,
        source_tokens_seen: Mapping[str, int] | None,
        validation_loss: float | None,
    ) -> Path:
        """Save one numbered resume point and apply numbered-file retention."""
        return self._save(
            path=self.directory / f"step_{step:06d}.pt",
            step=step,
            tokens_seen=tokens_seen,
            source_tokens_seen=source_tokens_seen,
            validation_loss=validation_loss,
            best_validation_loss=self.best_validation_loss,
            keep_last_n=self.keep_last_n,
        )

    def _save(
        self,
        *,
        path: Path,
        step: int,
        tokens_seen: int,
        source_tokens_seen: Mapping[str, int] | None,
        validation_loss: float | None,
        best_validation_loss: float | None,
        keep_last_n: int | None,
    ) -> Path:
        return save_checkpoint(
            path=path,
            model=self.model,
            optimizer=self.optimizer,
            step=step,
            training_config=self.training_config,
            wandb_run_id=self.wandb_run_id,
            wandb_project=self.wandb_project,
            wandb_entity=self.wandb_entity,
            keep_last_n=keep_last_n,
            run_config=self.run_config,
            data_position=step * self.sequences_per_step,
            tokens_seen=tokens_seen,
            source_tokens_seen=source_tokens_seen,
            validation_loss=validation_loss,
            best_validation_loss=best_validation_loss,
        )


@dataclass
class TrainingObserver:
    """Add source accounting, checkpointing, and W&B logging to trainer steps."""

    sampler: DeterministicMixtureSampler | None
    source_tokens_seen: dict[str, int] | None
    sequences_per_step: int
    tokens_per_sequence: int
    checkpoint_interval: int
    checkpoint_writer: CheckpointWriter
    logger: WandbEvaluationLogger

    def __call__(self, metrics: StepMetrics) -> None:
        logged_metrics = self._with_source_accounting(metrics)
        if logged_metrics.validation_loss is not None:
            self.checkpoint_writer.save_best_if_improved(
                step=logged_metrics.step,
                tokens_seen=logged_metrics.tokens_seen,
                source_tokens_seen=logged_metrics.source_tokens_seen,
                validation_loss=logged_metrics.validation_loss,
            )
        if logged_metrics.step % self.checkpoint_interval == 0:
            self.checkpoint_writer.save_step(
                step=logged_metrics.step,
                tokens_seen=logged_metrics.tokens_seen,
                source_tokens_seen=logged_metrics.source_tokens_seen,
                validation_loss=logged_metrics.validation_loss,
            )
        self.logger(logged_metrics)

    def _with_source_accounting(self, metrics: StepMetrics) -> StepMetrics:
        if self.sampler is None or self.source_tokens_seen is None:
            return metrics

        step_end_position = metrics.step * self.sequences_per_step
        step_start_position = step_end_position - self.sequences_per_step
        step_source_sequences = self.sampler.source_sequence_counts(
            step_start_position,
            step_end_position,
        )
        for source, sequence_count in step_source_sequences.items():
            self.source_tokens_seen[source] += sequence_count * self.tokens_per_sequence
        return replace(metrics, source_tokens_seen=dict(self.source_tokens_seen))
