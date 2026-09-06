"""Device selection and forward precision shared by training and evaluation."""

from contextlib import AbstractContextManager, nullcontext
from typing import Literal

import torch

Precision = Literal["fp32", "bf16"]
PrecisionArgument = Literal["auto", "fp32", "bf16"]


def resolve_device(requested: str) -> torch.device:
    """Select an available accelerator automatically or honor an explicit device."""
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is not available")
    return device


def validate_precision(device: torch.device | str, precision: str) -> torch.device:
    """Reject unsupported precision before allocating or running a forward pass."""
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


def resolve_precision(requested: str, device: torch.device) -> Precision:
    """Use BF16 automatically on supported CUDA devices and FP32 elsewhere."""
    if requested == "auto":
        return "bf16" if device.type == "cuda" and torch.cuda.is_bf16_supported() else "fp32"
    validate_precision(device, requested)
    return "bf16" if requested == "bf16" else "fp32"


def precision_context(device: torch.device | str, precision: str) -> AbstractContextManager[None]:
    resolved_device = validate_precision(device, precision)
    if precision == "fp32":
        return nullcontext()
    return torch.autocast(device_type=resolved_device.type, dtype=torch.bfloat16)
