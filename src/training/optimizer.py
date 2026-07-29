"""Optimizer construction for pretraining."""

import math
from numbers import Real

import torch


def build_adamw_parameter_groups(
    model: torch.nn.Module,
    weight_decay: float,
) -> list[dict[str, object]]:
    """Group trainable parameters by whether AdamW should decay them."""
    if (
        not isinstance(weight_decay, Real)
        or isinstance(weight_decay, bool)
        or not math.isfinite(float(weight_decay))
        or weight_decay < 0
    ):
        raise ValueError(f"weight_decay must be finite and non-negative, got {weight_decay!r}")

    decayed: list[torch.nn.Parameter] = []
    not_decayed: list[torch.nn.Parameter] = []
    seen: set[int] = set()
    for parameter in model.parameters():
        if not parameter.requires_grad or id(parameter) in seen:
            continue
        seen.add(id(parameter))
        if parameter.ndim >= 2:
            decayed.append(parameter)
        else:
            not_decayed.append(parameter)

    return [
        {"params": decayed, "weight_decay": float(weight_decay)},
        {"params": not_decayed, "weight_decay": 0.0},
    ]
