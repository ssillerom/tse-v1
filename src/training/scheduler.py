"""Learning-rate schedules used during pretraining."""

import math
from numbers import Real


def get_learning_rate(
    step: int,
    warmup_steps: int,
    max_steps: int,
    max_learning_rate: float,
    min_learning_rate: float,
) -> float:
    """Return the warmup-plus-cosine learning rate for one optimizer step."""
    if not isinstance(step, int) or isinstance(step, bool) or step < 0:
        raise ValueError(f"step must be a non-negative integer, got {step!r}")
    if not isinstance(warmup_steps, int) or isinstance(warmup_steps, bool) or warmup_steps < 0:
        raise ValueError(f"warmup_steps must be a non-negative integer, got {warmup_steps!r}")
    if not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps <= 0:
        raise ValueError(f"max_steps must be a positive integer, got {max_steps!r}")
    if warmup_steps > max_steps:
        raise ValueError(f"warmup_steps cannot exceed max_steps, got {warmup_steps} > {max_steps}")
    if (
        not isinstance(max_learning_rate, Real)
        or isinstance(max_learning_rate, bool)
        or not math.isfinite(float(max_learning_rate))
        or max_learning_rate <= 0
    ):
        raise ValueError(
            f"max_learning_rate must be finite and positive, got {max_learning_rate!r}"
        )
    if (
        not isinstance(min_learning_rate, Real)
        or isinstance(min_learning_rate, bool)
        or not math.isfinite(float(min_learning_rate))
        or min_learning_rate < 0
    ):
        raise ValueError(
            f"min_learning_rate must be finite and non-negative, got {min_learning_rate!r}"
        )
    if min_learning_rate > max_learning_rate:
        raise ValueError(
            "min_learning_rate cannot exceed max_learning_rate, "
            f"got {min_learning_rate} > {max_learning_rate}"
        )

    if step < warmup_steps:
        warmup_progress = step / warmup_steps
        return min_learning_rate + warmup_progress * (max_learning_rate - min_learning_rate)

    if step >= max_steps:
        return min_learning_rate

    decay_progress = (step - warmup_steps) / (max_steps - warmup_steps)
    cosine_multiplier = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
    return min_learning_rate + cosine_multiplier * (max_learning_rate - min_learning_rate)


def get_wsd_learning_rate(
    step: int,
    warmup_steps: int,
    decay_start_step: int,
    max_steps: int,
    max_learning_rate: float,
) -> float:
    """Return a warmup-stable-decay learning rate that finishes at zero."""
    if not isinstance(step, int) or isinstance(step, bool) or step < 0:
        raise ValueError(f"step must be a non-negative integer, got {step!r}")
    if not isinstance(warmup_steps, int) or isinstance(warmup_steps, bool) or warmup_steps < 0:
        raise ValueError(f"warmup_steps must be a non-negative integer, got {warmup_steps!r}")
    if (
        not isinstance(decay_start_step, int)
        or isinstance(decay_start_step, bool)
        or decay_start_step < 0
    ):
        raise ValueError(
            f"decay_start_step must be a non-negative integer, got {decay_start_step!r}"
        )
    if not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps <= 0:
        raise ValueError(f"max_steps must be a positive integer, got {max_steps!r}")
    if warmup_steps > decay_start_step:
        raise ValueError(
            f"warmup_steps cannot exceed decay_start_step, got {warmup_steps} > {decay_start_step}"
        )
    if decay_start_step > max_steps:
        raise ValueError(
            f"decay_start_step cannot exceed max_steps, got {decay_start_step} > {max_steps}"
        )
    if (
        not isinstance(max_learning_rate, Real)
        or isinstance(max_learning_rate, bool)
        or not math.isfinite(float(max_learning_rate))
        or max_learning_rate <= 0
    ):
        raise ValueError(
            f"max_learning_rate must be finite and positive, got {max_learning_rate!r}"
        )

    if step < warmup_steps:
        return max_learning_rate * step / warmup_steps
    if step < decay_start_step:
        return max_learning_rate
    if step >= max_steps:
        return 0.0
    if decay_start_step == max_steps:
        return max_learning_rate
    if decay_start_step == max_steps - 1:
        return 0.0

    decay_progress = (step - decay_start_step) / (max_steps - decay_start_step - 1)
    return max_learning_rate * 0.5 * (1.0 + math.cos(math.pi * decay_progress))
