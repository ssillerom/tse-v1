"""Training loop primitives for the V1 language model."""

import math
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import islice
from numbers import Real
from typing import Literal

import torch

from src.model.gpt import GPT, IGNORE_INDEX
from src.runtime import Precision as Precision
from src.runtime import precision_context, validate_precision

from .evaluation import perplexity_from_loss
from .rng import capture_torch_rng_state, restore_torch_rng_state
from .scheduler import get_learning_rate, get_wsd_learning_rate

Batch = tuple[torch.Tensor, torch.Tensor]
LearningRateSchedule = Literal["cosine", "wsd"]


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
        # Validate schedule-specific fields and the shared numeric boundaries.
        self.learning_rate_at(0)

    def learning_rate_at(self, step: int) -> float:
        """Return this run's scheduled rate for a zero-based optimizer step."""
        if self.learning_rate_schedule == "cosine":
            if self.decay_start_step is not None:
                raise ValueError("decay_start_step is supported only by the wsd schedule")
            return get_learning_rate(
                step=step,
                warmup_steps=self.warmup_steps,
                max_steps=self.max_steps,
                max_learning_rate=self.max_learning_rate,
                min_learning_rate=self.min_learning_rate,
            )
        if self.learning_rate_schedule == "wsd":
            if self.decay_start_step is None:
                raise ValueError("decay_start_step is required for the wsd schedule")
            if self.min_learning_rate != 0.0:
                raise ValueError("min_learning_rate must be 0.0 for the wsd schedule")
            return get_wsd_learning_rate(
                step=step,
                warmup_steps=self.warmup_steps,
                decay_start_step=self.decay_start_step,
                max_steps=self.max_steps,
                max_learning_rate=self.max_learning_rate,
            )
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
    source_tokens_seen: dict[str, int] | None = None
    validation_loss: float | None = None
    validation_perplexity: float | None = None
    validation_domains: dict[str, "EvaluationMetrics"] | None = None

    def __post_init__(self) -> None:
        if (self.validation_loss is None) != (self.validation_perplexity is None):
            raise ValueError(
                "validation_loss and validation_perplexity must either both be present "
                "or both be absent"
            )
        if self.validation_domains is not None and self.validation_loss is None:
            raise ValueError("validation_domains require aggregate validation metrics")
        if self.source_tokens_seen is not None:
            if any(
                not isinstance(source_name, str)
                or not source_name
                or not isinstance(token_count, int)
                or isinstance(token_count, bool)
                or token_count < 0
                for source_name, token_count in self.source_tokens_seen.items()
            ):
                raise ValueError(
                    "source_tokens_seen must map source names to non-negative integers"
                )
            if sum(self.source_tokens_seen.values()) != self.tokens_seen:
                raise ValueError("source_tokens_seen must sum to tokens_seen")


StepCallback = Callable[[StepMetrics], None]


@dataclass(frozen=True)
class EvaluationMetrics:
    """Token-weighted metrics for one validation distribution."""

    loss: float
    perplexity: float
    target_tokens: int


@dataclass(frozen=True)
class DomainEvaluation:
    """Per-domain validation metrics plus a recipe-weighted aggregate."""

    loss: float
    perplexity: float
    target_tokens: int
    domains: dict[str, EvaluationMetrics]


def evaluate_metrics(
    model: GPT,
    batches: Iterable[Batch],
    device: torch.device | str,
    max_batches: int | None = None,
    precision: Precision = "fp32",
) -> EvaluationMetrics:
    """Return token-weighted metrics without changing the caller's model mode."""
    validate_precision(device, precision)
    if max_batches is not None and (
        not isinstance(max_batches, int) or isinstance(max_batches, bool) or max_batches <= 0
    ):
        raise ValueError(f"max_batches must be a positive integer, got {max_batches!r}")

    was_training = model.training
    rng_snapshot = capture_torch_rng_state()
    model.eval()
    total_loss = 0.0
    total_targets = 0

    try:
        with torch.no_grad():
            for input_ids, targets in islice(batches, max_batches):
                target_count = int((targets != IGNORE_INDEX).sum().item())
                if target_count == 0:
                    continue
                input_ids = input_ids.to(device)
                targets = targets.to(device)
                with precision_context(device, precision):
                    _, loss = model(input_ids, targets)
                if loss is None:
                    raise RuntimeError("model did not return a loss for an evaluation batch")
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"evaluation loss is not finite: {loss.item()}")

                total_loss += loss.item() * target_count
                total_targets += target_count
    finally:
        # Creating a DataLoader iterator consumes PyTorch's global RNG even
        # when validation is not shuffled. Evaluation is observational, so it
        # must not change later dropout masks or worker seeds in training.
        try:
            model.train(was_training)
        finally:
            restore_torch_rng_state(rng_snapshot)

    if total_targets == 0:
        raise ValueError("evaluation batches contain no target tokens")
    loss = total_loss / total_targets
    return EvaluationMetrics(
        loss=loss,
        perplexity=perplexity_from_loss(loss),
        target_tokens=total_targets,
    )


def evaluate(
    model: GPT,
    batches: Iterable[Batch],
    device: torch.device | str,
    max_batches: int | None = None,
    precision: Precision = "fp32",
) -> float:
    """Return token-weighted mean loss without changing the caller's model mode."""
    return evaluate_metrics(
        model=model,
        batches=batches,
        device=device,
        max_batches=max_batches,
        precision=precision,
    ).loss


def evaluate_domains(
    model: GPT,
    batches_by_domain: Mapping[str, Iterable[Batch]],
    domain_weights: Mapping[str, float],
    device: torch.device | str,
    max_batches: int | None = None,
    precision: Precision = "fp32",
) -> DomainEvaluation:
    """Evaluate fixed domains and combine their losses using recipe weights."""
    if not batches_by_domain:
        raise ValueError("batches_by_domain must not be empty")
    if set(domain_weights) != set(batches_by_domain):
        raise ValueError("domain_weights keys must match validation domains")

    total_weight = 0.0
    for domain, weight in domain_weights.items():
        if (
            not isinstance(weight, Real)
            or isinstance(weight, bool)
            or not math.isfinite(float(weight))
            or weight <= 0
        ):
            raise ValueError(
                f"domain weight for {domain!r} must be finite and positive, got {weight!r}"
            )
        total_weight += float(weight)

    domains = {
        domain: evaluate_metrics(
            model=model,
            batches=batches,
            device=device,
            max_batches=max_batches,
            precision=precision,
        )
        for domain, batches in batches_by_domain.items()
    }
    aggregate_loss = sum(
        domains[domain].loss * float(domain_weights[domain]) / total_weight for domain in domains
    )
    return DomainEvaluation(
        loss=aggregate_loss,
        perplexity=perplexity_from_loss(aggregate_loss),
        target_tokens=sum(metrics.target_tokens for metrics in domains.values()),
        domains=domains,
    )


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
    validation_batches: Iterable[Batch] | Mapping[str, Iterable[Batch]] | None = None,
    validation_weights: Mapping[str, float] | None = None,
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
    if isinstance(validation_batches, Mapping):
        if not validation_batches:
            raise ValueError("validation_batches must not be empty")
        if validation_weights is None:
            raise ValueError("validation_weights are required for domain validation")
        if any(isinstance(batches, Iterator) for batches in validation_batches.values()):
            raise ValueError("validation batches must be reiterable, not one-shot iterators")
    else:
        if validation_weights is not None:
            raise ValueError("validation_weights require domain validation batches")
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

        learning_rate = config.learning_rate_at(step_index)
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
        validation_domains = None
        if (
            validation_batches is not None
            and config.eval_interval is not None
            # Always evaluate the segment boundary so a deliberate stop or
            # final run produces a comparable checkpoint.
            and (completed_step % config.eval_interval == 0 or completed_step == effective_end_step)
        ):
            if isinstance(validation_batches, Mapping):
                if validation_weights is None:
                    raise RuntimeError("validated domain evaluation has no weights")
                domain_evaluation = evaluate_domains(
                    model=model,
                    batches_by_domain=validation_batches,
                    domain_weights=validation_weights,
                    device=device,
                    max_batches=config.eval_batches,
                    precision=config.precision,
                )
                validation_loss = domain_evaluation.loss
                validation_perplexity = domain_evaluation.perplexity
                validation_domains = domain_evaluation.domains
            else:
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
            validation_domains=validation_domains,
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
    validate_precision(device, precision)
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
        with precision_context(device, precision):
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
