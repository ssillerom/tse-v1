"""Atomic checkpoints for resumable V1 training."""

import os
from dataclasses import asdict
from pathlib import Path
from typing import cast

import torch

from src.model.gpt import GPT

from .trainer import TrainingConfig

CHECKPOINT_FORMAT_VERSION = 1


def save_checkpoint(
    path: str | Path,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    step: int,
    training_config: TrainingConfig,
) -> Path:
    """Atomically save the state required to resume training."""
    if not isinstance(step, int) or isinstance(step, bool) or step < 0:
        raise ValueError(f"step must be a non-negative integer, got {step!r}")
    if step > training_config.max_steps:
        raise ValueError(
            "step cannot exceed training_config.max_steps, "
            f"got {step} > {training_config.max_steps}"
        )

    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = checkpoint_path.with_name(f"{checkpoint_path.name}.tmp")
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "step": step,
        "torch_rng_state": torch.random.get_rng_state(),
        "cuda_rng_states": (torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),
        "mps_rng_state": (torch.mps.get_rng_state() if torch.backends.mps.is_available() else None),
        "model_config": asdict(model.config),
        "training_config": asdict(training_config),
    }

    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, checkpoint_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    return checkpoint_path


def load_checkpoint(
    path: str | Path,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    training_config: TrainingConfig,
    map_location: torch.device | str = "cpu",
) -> int:
    """Restore model, optimizer, and random state, then return the completed step."""
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    payload = torch.load(
        checkpoint_path,
        map_location=map_location,
        weights_only=True,
    )
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload must be a dictionary")
    if payload.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported checkpoint format_version: {payload.get('format_version')!r}"
        )

    step = payload.get("step")
    if not isinstance(step, int) or isinstance(step, bool) or step < 0:
        raise ValueError(f"checkpoint contains an invalid step: {step!r}")

    model_state = payload.get("model_state")
    optimizer_state = payload.get("optimizer_state")
    torch_rng_state = payload.get("torch_rng_state")
    model_config = payload.get("model_config")
    saved_training_config = payload.get("training_config")
    if not isinstance(model_state, dict):
        raise ValueError("checkpoint contains an invalid model_state")
    if not isinstance(optimizer_state, dict):
        raise ValueError("checkpoint contains an invalid optimizer_state")
    if not isinstance(torch_rng_state, torch.Tensor):
        raise ValueError("checkpoint contains an invalid torch_rng_state")
    if not isinstance(model_config, dict):
        raise ValueError("checkpoint contains an invalid model_config")
    if not isinstance(saved_training_config, dict):
        raise ValueError("checkpoint contains an invalid training_config")
    if model_config != asdict(model.config):
        raise ValueError("checkpoint model_config does not match the current model configuration")
    if saved_training_config != asdict(training_config):
        raise ValueError(
            "checkpoint training_config does not match the current training configuration"
        )
    if step > training_config.max_steps:
        raise ValueError(
            "checkpoint step cannot exceed training_config.max_steps, "
            f"got {step} > {training_config.max_steps}"
        )

    model.load_state_dict(cast(dict[str, torch.Tensor], model_state))
    optimizer.load_state_dict(cast(dict[str, object], optimizer_state))
    torch.random.set_rng_state(torch_rng_state.cpu())

    cuda_rng_states = payload.get("cuda_rng_states")
    if cuda_rng_states is not None and torch.cuda.is_available():
        if not isinstance(cuda_rng_states, list) or not all(
            isinstance(state, torch.Tensor) for state in cuda_rng_states
        ):
            raise ValueError("checkpoint contains invalid cuda_rng_states")
        torch.cuda.set_rng_state_all(cuda_rng_states)

    mps_rng_state = payload.get("mps_rng_state")
    if mps_rng_state is not None and torch.backends.mps.is_available():
        if not isinstance(mps_rng_state, torch.Tensor):
            raise ValueError("checkpoint contains an invalid mps_rng_state")
        torch.mps.set_rng_state(mps_rng_state.cpu())

    return step
