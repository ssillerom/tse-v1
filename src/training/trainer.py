"""Training loop primitives for the V1 language model."""

import math
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from numbers import Real
from typing import Literal

import torch

from src.model.gpt import GPT, IGNORE_INDEX

from .evaluation import perplexity_from_loss
from .rng import capture_torch_rng_state, restore_torch_rng_state
from .scheduler import get_learning_rate, get_wsd_learning_rate

Batch = tuple[torch.Tensor, torch.Tensor]
Precision = Literal["fp32", "bf16"]
LearningRateSchedule = Literal["cosine", "wsd"]


def _validate_max_grad_norm(max_grad_norm: object) -> None:
    if (
        not isinstance(max_grad_norm, Real)
        or isinstance(max_grad_norm, bool)
        or not math.isfinite(float(max_grad_norm))
        or max_grad_norm <= 0
    ):
        raise ValueError(f"max_grad_norm must be finite and positive, got {max_grad_norm!r}")


def _validate_precision_device(
    device: torch.device | str,
    precision: Precision,
) -> torch.device:
    resolved_device = torch.device(device)
    if precision not in ("fp32", "bf16"):
        raise ValueError(f"precision must be one of ('fp32', 'bf16'), got {precision!r}")
    if precision == "bf16" and resolved_device.type not in ("cpu", "cuda"):
        raise ValueError(
            f"bf16 precision is supported only on CPU or CUDA, got device {resolved_device.type!r}"
        )
    if (
        precision == "bf16"
        and resolved_device.type == "cuda"
        and not torch.cuda.is_bf16_supported()
    ):
        raise ValueError("bf16 precision is not supported by the requested CUDA device")
    return resolved_device


def _forward_precision_context(
    device: torch.device | str,
    precision: Precision,
) -> AbstractContextManager[None]:
    resolved_device = _validate_precision_device(device, precision)
    if precision == "fp32":
        return nullcontext()
    return torch.autocast(device_type=resolved_device.type, dtype=torch.bfloat16)


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for a finite pretraining run."""

    max_steps: int
    grad_accum_steps: int = 1
    warmup_steps: int = 0
    max_learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    max_grad_norm: float = 1.0
    eval_interval: int | None = None
    eval_batches: int | None = None
    precision: Precision = "fp32"
    learning_rate_schedule: LearningRateSchedule = "cosine"
    decay_start_step: int | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.grad_accum_steps, int)
            or isinstance(self.grad_accum_steps, bool)
            or self.grad_accum_steps <= 0
        ):
            raise ValueError(
                f"grad_accum_steps must be a positive integer, got {self.grad_accum_steps!r}"
            )
        _validate_max_grad_norm(self.max_grad_norm)
        if self.eval_interval is not None and (
            not isinstance(self.eval_interval, int)
            or isinstance(self.eval_interval, bool)
            or self.eval_interval <= 0
        ):
            raise ValueError(
                f"eval_interval must be a positive integer, got {self.eval_interval!r}"
            )
        if self.eval_batches is not None and (
            not isinstance(self.eval_batches, int)
            or isinstance(self.eval_batches, bool)
            or self.eval_batches <= 0
        ):
            raise ValueError(f"eval_batches must be a positive integer, got {self.eval_batches!r}")
        if self.eval_batches is not None and self.eval_interval is None:
            raise ValueError("eval_batches requires eval_interval")
        if self.precision not in ("fp32", "bf16"):
            raise ValueError(f"precision must be one of ('fp32', 'bf16'), got {self.precision!r}")
        if self.learning_rate_schedule == "cosine":
            if self.decay_start_step is not None:
                raise ValueError("decay_start_step is supported only by the wsd schedule")
            get_learning_rate(
                step=0,
                warmup_steps=self.warmup_steps,
                max_steps=self.max_steps,
                max_learning_rate=self.max_learning_rate,
                min_learning_rate=self.min_learning_rate,
            )
        elif self.learning_rate_schedule == "wsd":
            if self.decay_start_step is None:
                raise ValueError("decay_start_step is required for the wsd schedule")
            if self.min_learning_rate != 0.0:
                raise ValueError("min_learning_rate must be 0.0 for the wsd schedule")
            get_wsd_learning_rate(
                step=0,
                warmup_steps=self.warmup_steps,
                decay_start_step=self.decay_start_step,
                max_steps=self.max_steps,
                max_learning_rate=self.max_learning_rate,
            )
        else:
            raise ValueError(
                "learning_rate_schedule must be one of ('cosine', 'wsd'), "
                f"got {self.learning_rate_schedule!r}"
            )


@dataclass(frozen=True)
class StepMetrics:
    """Observable metrics produced by one completed optimizer update."""

    step: int
    loss: float
    gradient_norm: float
    learning_rate: float
    tokens_in_step: int
    tokens_seen: int
    step_time_seconds: float
    tokens_per_second: float
    validation_loss: float | None = None
    validation_perplexity: float | None = None

    def __post_init__(self) -> None:
        if (self.validation_loss is None) != (self.validation_perplexity is None):
            raise ValueError(
                "validation_loss and validation_perplexity must either both be present "
                "or both be absent"
            )


StepCallback = Callable[[StepMetrics], None]


def evaluate(
    model: GPT,
    batches: Iterable[Batch],
    device: torch.device | str,
    max_batches: int | None = None,
    precision: Precision = "fp32",
) -> float:
    """Return token-weighted mean loss without changing the caller's model mode."""
    _validate_precision_device(device, precision)
    if max_batches is not None and (
        not isinstance(max_batches, int) or isinstance(max_batches, bool) or max_batches <= 0
    ):
        raise ValueError(f"max_batches must be a positive integer, got {max_batches!r}")

    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_targets = 0

    try:
        with torch.no_grad():
            for batch_index, (input_ids, targets) in enumerate(batches):
                if max_batches is not None and batch_index >= max_batches:
                    break

                target_count = int((targets != IGNORE_INDEX).sum().item())
                if target_count == 0:
                    continue
                input_ids = input_ids.to(device)
                targets = targets.to(device)
                with _forward_precision_context(device, precision):
                    _, loss = model(input_ids, targets)
                if loss is None:
                    raise RuntimeError("model did not return a loss for an evaluation batch")
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"evaluation loss is not finite: {loss.item()}")

                total_loss += loss.item() * target_count
                total_targets += target_count
    finally:
        model.train(was_training)

    if total_targets == 0:
        raise ValueError("evaluation batches contain no target tokens")
    return total_loss / total_targets


def _next_batch(
    batches: Iterable[Batch],
    batch_iterator: Iterator[Batch],
) -> tuple[Batch, Iterator[Batch]]:
    try:
        return next(batch_iterator), batch_iterator
    except StopIteration:
        restarted_iterator = iter(batches)
        try:
            return next(restarted_iterator), restarted_iterator
        except StopIteration as error:
            raise ValueError("train_batches must contain at least one batch") from error


def _batch_iterator_at_step(
    batches: Iterable[Batch],
    completed_steps: int,
    grad_accum_steps: int,
    batches_start_step: int,
    tokens_seen_at_start: int,
) -> tuple[Iterator[Batch], int]:
    steps_to_replay = completed_steps - batches_start_step
    rng_snapshot = capture_torch_rng_state()
    try:
        batch_iterator = iter(batches)
        tokens_seen = tokens_seen_at_start
        for _ in range(steps_to_replay * grad_accum_steps):
            batch, batch_iterator = _next_batch(batches, batch_iterator)
            tokens_seen += int((batch[1] != IGNORE_INDEX).sum().item())
        return batch_iterator, tokens_seen
    finally:
        restore_torch_rng_state(rng_snapshot)


def _synchronize_device(device: torch.device | str) -> None:
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda":
        torch.cuda.synchronize(resolved_device)
    elif resolved_device.type == "mps":
        torch.mps.synchronize()


def train(
    model: GPT,
    optimizer: torch.optim.Optimizer,
    train_batches: Iterable[Batch],
    config: TrainingConfig,
    device: torch.device | str,
    validation_batches: Iterable[Batch] | None = None,
    start_step: int = 0,
    end_step: int | None = None,
    on_step: StepCallback | None = None,
    batches_start_step: int = 0,
    tokens_seen_at_start: int = 0,
) -> tuple[StepMetrics, ...]:
    """Train one deterministic segment and return metrics for completed steps.

    A non-zero ``start_step`` rebuilds the batch position by replaying a
    deterministic, reiterable source from its beginning.
    """
    if (
        not isinstance(start_step, int)
        or isinstance(start_step, bool)
        or not 0 <= start_step <= config.max_steps
    ):
        raise ValueError(
            f"start_step must be between 0 and max_steps={config.max_steps}, got {start_step!r}"
        )
    if (
        not isinstance(batches_start_step, int)
        or isinstance(batches_start_step, bool)
        or not 0 <= batches_start_step <= start_step
    ):
        raise ValueError(
            f"batches_start_step must be between 0 and start_step, got {batches_start_step!r}"
        )
    if (
        not isinstance(tokens_seen_at_start, int)
        or isinstance(tokens_seen_at_start, bool)
        or tokens_seen_at_start < 0
    ):
        raise ValueError("tokens_seen_at_start must be a non-negative integer")
    effective_end_step = config.max_steps if end_step is None else end_step
    if (
        not isinstance(effective_end_step, int)
        or isinstance(effective_end_step, bool)
        or not start_step <= effective_end_step <= config.max_steps
    ):
        raise ValueError(
            "end_step must be between "
            f"start_step={start_step} and max_steps={config.max_steps}, "
            f"got {effective_end_step!r}"
        )
    if config.eval_interval is not None and validation_batches is None:
        raise ValueError("validation_batches are required when eval_interval is configured")
    if isinstance(train_batches, Iterator):
        raise ValueError("train_batches must be reiterable, not a one-shot iterator")
    if validation_batches is not None and isinstance(validation_batches, Iterator):
        raise ValueError("validation_batches must be reiterable, not a one-shot iterator")

    history: list[StepMetrics] = []
    if start_step == effective_end_step:
        return ()
    batch_iterator, tokens_seen = _batch_iterator_at_step(
        batches=train_batches,
        completed_steps=start_step,
        grad_accum_steps=config.grad_accum_steps,
        batches_start_step=batches_start_step,
        tokens_seen_at_start=tokens_seen_at_start,
    )

    for step_index in range(start_step, effective_end_step):
        microbatches: list[Batch] = []
        while len(microbatches) < config.grad_accum_steps:
            batch, batch_iterator = _next_batch(train_batches, batch_iterator)
            microbatches.append(batch)

        if config.learning_rate_schedule == "wsd":
            if config.decay_start_step is None:
                raise RuntimeError("validated WSD config has no decay_start_step")
            learning_rate = get_wsd_learning_rate(
                step=step_index,
                warmup_steps=config.warmup_steps,
                decay_start_step=config.decay_start_step,
                max_steps=config.max_steps,
                max_learning_rate=config.max_learning_rate,
            )
        else:
            learning_rate = get_learning_rate(
                step=step_index,
                warmup_steps=config.warmup_steps,
                max_steps=config.max_steps,
                max_learning_rate=config.max_learning_rate,
                min_learning_rate=config.min_learning_rate,
            )
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = learning_rate

        tokens_in_step = sum(
            int((targets != IGNORE_INDEX).sum().item()) for _, targets in microbatches
        )
        _synchronize_device(device)
        step_started_at = time.perf_counter()
        loss, gradient_norm = train_step(
            model=model,
            optimizer=optimizer,
            microbatches=microbatches,
            max_grad_norm=config.max_grad_norm,
            device=device,
            precision=config.precision,
        )
        _synchronize_device(device)
        step_time_seconds = time.perf_counter() - step_started_at
        tokens_seen += tokens_in_step
        tokens_per_second = tokens_in_step / max(step_time_seconds, 1e-9)
        completed_step = step_index + 1
        validation_loss = None
        validation_perplexity = None
        if (
            validation_batches is not None
            and config.eval_interval is not None
            and completed_step % config.eval_interval == 0
        ):
            validation_loss = evaluate(
                model=model,
                batches=validation_batches,
                device=device,
                max_batches=config.eval_batches,
                precision=config.precision,
            )
            validation_perplexity = perplexity_from_loss(validation_loss)

        metrics = StepMetrics(
            step=completed_step,
            loss=loss,
            gradient_norm=gradient_norm,
            learning_rate=learning_rate,
            tokens_in_step=tokens_in_step,
            tokens_seen=tokens_seen,
            step_time_seconds=step_time_seconds,
            tokens_per_second=tokens_per_second,
            validation_loss=validation_loss,
            validation_perplexity=validation_perplexity,
        )
        history.append(metrics)
        if on_step is not None:
            on_step(metrics)

    return tuple(history)


def train_step(
    model: GPT,
    optimizer: torch.optim.Optimizer,
    microbatches: Sequence[Batch],
    max_grad_norm: float,
    device: torch.device | str,
    precision: Precision = "fp32",
) -> tuple[float, float]:
    """Run one optimizer update over one or more accumulated microbatches."""
    _validate_precision_device(device, precision)
    if not microbatches:
        raise ValueError("microbatches must contain at least one batch")
    _validate_max_grad_norm(max_grad_norm)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    accumulated_loss = 0.0
    target_counts = [int((targets != IGNORE_INDEX).sum().item()) for _, targets in microbatches]
    total_targets = sum(target_counts)
    if total_targets == 0:
        raise ValueError("microbatches contain no target tokens")

    for (input_ids, targets), target_count in zip(
        microbatches,
        target_counts,
        strict=True,
    ):
        if target_count == 0:
            continue
        input_ids = input_ids.to(device)
        targets = targets.to(device)
        with _forward_precision_context(device, precision):
            _, loss = model(input_ids, targets)
        if loss is None:
            raise RuntimeError("model did not return a loss for a training batch")
        if not torch.isfinite(loss):
            raise FloatingPointError(f"training loss is not finite: {loss.item()}")

        loss_weight = target_count / total_targets
        (loss * loss_weight).backward()
        accumulated_loss += loss.item() * loss_weight

    gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=max_grad_norm,
    )
    gradient_norm_value = gradient_norm.item()
    if not math.isfinite(gradient_norm_value):
        raise FloatingPointError(f"gradient norm is not finite: {gradient_norm_value}")

    optimizer.step()
    return accumulated_loss, gradient_norm_value
