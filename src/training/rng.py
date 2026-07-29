"""Capture and restore PyTorch random state without changing device availability."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TorchRngSnapshot:
    """CPU and available accelerator random-generator states."""

    cpu_state: torch.Tensor
    cuda_states: list[torch.Tensor] | None
    mps_state: torch.Tensor | None


def capture_torch_rng_state() -> TorchRngSnapshot:
    """Return the current PyTorch RNG state for every available device type."""
    return TorchRngSnapshot(
        cpu_state=torch.random.get_rng_state(),
        cuda_states=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        mps_state=torch.mps.get_rng_state() if torch.backends.mps.is_available() else None,
    )


def restore_torch_rng_state(snapshot: TorchRngSnapshot) -> None:
    """Restore a snapshot on the device types available in this process."""
    torch.random.set_rng_state(snapshot.cpu_state.cpu())
    if snapshot.cuda_states is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([state.cpu() for state in snapshot.cuda_states])
    if snapshot.mps_state is not None and torch.backends.mps.is_available():
        torch.mps.set_rng_state(snapshot.mps_state.cpu())
