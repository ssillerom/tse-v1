"""Training loop primitives for the V1 language model."""

import math
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from numbers import Real

import torch

from src.model.gpt import GPT, IGNORE_INDEX

from .scheduler import get_learning_rate

Batch = tuple[torch.Tensor, torch.Tensor]


def _validate_max_grad_norm(max_grad_norm: object) -> None:
    if (
        not isinstance(max_grad_norm, Real)
        or isinstance(max_grad_norm, bool)
        or not math.isfinite(float(max_grad_norm))
        or max_grad_norm <= 0
    ):
        raise ValueError(f"max_grad_norm must be finite and positive, got {max_grad_norm!r}")


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

        get_learning_rate(
            step=0,
            warmup_steps=self.warmup_steps,
            max_steps=self.max_steps,
            max_learning_rate=self.max_learning_rate,
            min_learning_rate=self.min_learning_rate,
        )


@dataclass(frozen=True)
class StepMetrics:
    """Observable metrics produced by one completed optimizer update."""

    step: int
    loss: float
    gradient_norm: float
    learning_rate: float
    validation_loss: float | None = None


def evaluate(
    model: GPT,
    batches: Iterable[Batch],
    device: torch.device | str,
    max_batches: int | None = None,
) -> float:
    """Return token-weighted mean loss without changing the caller's model mode."""
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
) -> Iterator[Batch]:
    if completed_steps == 0:
        return iter(batches)

    cpu_rng_state = torch.random.get_rng_state()
    cuda_rng_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    mps_rng_state = torch.mps.get_rng_state() if torch.backends.mps.is_available() else None
    try:
        batch_iterator = iter(batches)
        for _ in range(completed_steps * grad_accum_steps):
            _, batch_iterator = _next_batch(batches, batch_iterator)
        return batch_iterator
    finally:
        torch.random.set_rng_state(cpu_rng_state)
        if cuda_rng_states is not None:
            torch.cuda.set_rng_state_all(cuda_rng_states)
        if mps_rng_state is not None:
            torch.mps.set_rng_state(mps_rng_state)


def train(
    model: GPT,
    optimizer: torch.optim.Optimizer,
    train_batches: Iterable[Batch],
    config: TrainingConfig,
    device: torch.device | str,
    validation_batches: Iterable[Batch] | None = None,
    start_step: int = 0,
    end_step: int | None = None,
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
    batch_iterator = _batch_iterator_at_step(
        batches=train_batches,
        completed_steps=start_step,
        grad_accum_steps=config.grad_accum_steps,
    )

    for step_index in range(start_step, effective_end_step):
        microbatches: list[Batch] = []
        while len(microbatches) < config.grad_accum_steps:
            batch, batch_iterator = _next_batch(train_batches, batch_iterator)
            microbatches.append(batch)

        learning_rate = get_learning_rate(
            step=step_index,
            warmup_steps=config.warmup_steps,
            max_steps=config.max_steps,
            max_learning_rate=config.max_learning_rate,
            min_learning_rate=config.min_learning_rate,
        )
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = learning_rate

        loss, gradient_norm = train_step(
            model=model,
            optimizer=optimizer,
            microbatches=microbatches,
            max_grad_norm=config.max_grad_norm,
            device=device,
        )
        completed_step = step_index + 1
        validation_loss = None
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
            )

        history.append(
            StepMetrics(
                step=completed_step,
                loss=loss,
                gradient_norm=gradient_norm,
                learning_rate=learning_rate,
                validation_loss=validation_loss,
            )
        )

    return tuple(history)


def train_step(
    model: GPT,
    optimizer: torch.optim.Optimizer,
    microbatches: Sequence[Batch],
    max_grad_norm: float,
    device: torch.device | str,
) -> tuple[float, float]:
    """Run one optimizer update over one or more accumulated microbatches."""
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
