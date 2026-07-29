"""Weights & Biases logging for V1 training and evaluation."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import tiktoken
import torch
import wandb

from src.model.gpt import GPT

from .evaluation import (
    EVALUATION_PROMPTS,
    MAX_EVALUATION_NEW_TOKENS,
    EvaluationPrompt,
    generate_prompt_samples,
)
from .trainer import StepMetrics


class WandbRun(Protocol):
    """The small part of a W&B run required by this logger."""

    id: str

    def log(self, data: dict[str, object]) -> None: ...

    def define_metric(
        self,
        name: str,
        *,
        step_metric: str | None = None,
        summary: str | None = None,
        hidden: bool | None = None,
    ) -> object: ...


def metrics_to_wandb(metrics: StepMetrics) -> dict[str, float | int]:
    """Map one optimizer step to stable W&B metric names."""
    payload: dict[str, float | int] = {
        "trainer/global_step": metrics.step,
        "train/loss": metrics.loss,
        "optimizer/gradient_norm": metrics.gradient_norm,
        "optimizer/learning_rate": metrics.learning_rate,
        "trainer/tokens_in_step": metrics.tokens_in_step,
        "trainer/tokens_seen": metrics.tokens_seen,
        "performance/step_time_seconds": metrics.step_time_seconds,
        "performance/tokens_per_second": metrics.tokens_per_second,
    }
    if metrics.validation_loss is not None:
        if metrics.validation_perplexity is None:
            raise ValueError("validation_perplexity is required when validation_loss is present")
        payload["validation/loss"] = metrics.validation_loss
        payload["validation/perplexity"] = metrics.validation_perplexity
    return payload


@dataclass
class WandbEvaluationLogger:
    """Log scalar training metrics and periodic fixed-prompt generations."""

    run: WandbRun
    model: GPT
    encoding: tiktoken.Encoding
    device: torch.device | str
    sample_interval: int = 100
    max_new_tokens: int = 64
    prompts: Sequence[EvaluationPrompt] = EVALUATION_PROMPTS

    def __post_init__(self) -> None:
        if (
            not isinstance(self.sample_interval, int)
            or isinstance(self.sample_interval, bool)
            or self.sample_interval <= 0
        ):
            raise ValueError(
                f"sample_interval must be a positive integer, got {self.sample_interval!r}"
            )
        if (
            not isinstance(self.max_new_tokens, int)
            or isinstance(self.max_new_tokens, bool)
            or self.max_new_tokens <= 0
            or self.max_new_tokens > MAX_EVALUATION_NEW_TOKENS
        ):
            raise ValueError(
                f"max_new_tokens must be a positive integer of at most "
                f"{MAX_EVALUATION_NEW_TOKENS}, got {self.max_new_tokens!r}"
            )
        self.run.define_metric("trainer/global_step", hidden=True)
        self.run.define_metric("train/*", step_metric="trainer/global_step")
        self.run.define_metric("optimizer/*", step_metric="trainer/global_step")
        self.run.define_metric("trainer/tokens_*", step_metric="trainer/global_step")
        self.run.define_metric("performance/*", step_metric="trainer/global_step")
        self.run.define_metric(
            "validation/loss",
            step_metric="trainer/global_step",
            summary="min",
        )
        self.run.define_metric(
            "validation/perplexity",
            step_metric="trainer/global_step",
            summary="min",
        )
        self.run.define_metric(
            "samples/fixed_prompts",
            step_metric="trainer/global_step",
        )

    def __call__(self, metrics: StepMetrics) -> None:
        scalar_payload: dict[str, object] = dict(metrics_to_wandb(metrics))
        self.run.log(scalar_payload)
        if metrics.step % self.sample_interval != 0:
            return
        self.log_samples(metrics.step)

    def log_samples(self, step: int) -> None:
        """Generate and log the complete fixed-prompt suite at one step."""
        samples = generate_prompt_samples(
            model=self.model,
            encoding=self.encoding,
            prompts=self.prompts,
            max_new_tokens=self.max_new_tokens,
            device=self.device,
        )
        table = wandb.Table(
            columns=["step", "category", "prompt", "continuation"],
            data=[
                [step, sample.category, sample.prompt, sample.continuation] for sample in samples
            ],
        )
        self.run.log(
            {
                "trainer/global_step": step,
                "samples/fixed_prompts": table,
            }
        )
