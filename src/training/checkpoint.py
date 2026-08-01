"""Atomic checkpoints for resumable V1 training."""

import math
import os
import re
import warnings
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from numbers import Real
from pathlib import Path
from typing import cast

import torch

from src.model.gpt import GPT

from .rng import TorchRngSnapshot, capture_torch_rng_state, restore_torch_rng_state
from .trainer import TrainingConfig

CHECKPOINT_FORMAT_VERSION = 6
SUPPORTED_CHECKPOINT_FORMAT_VERSIONS = frozenset({1, 2, 3, 4, 5, CHECKPOINT_FORMAT_VERSION})
CHECKPOINT_FILENAME_PATTERN = re.compile(r"step_(\d+)\.pt\Z")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class RestoredCheckpoint:
    """State recovered from one validated checkpoint."""

    path: Path
    step: int
    wandb_run_id: str | None
    wandb_project: str | None
    wandb_entity: str | None
    data_position: int
    tokens_seen: int
    source_tokens_seen: dict[str, int] | None
    validation_loss: float | None
    best_validation_loss: float | None


@dataclass(frozen=True)
class TrainingRunConfig:
    """Settings outside TrainingConfig that determine exact continuation."""

    data_contract_sha256: str
    seed: int
    batch_size: int
    optimizer_name: str
    optimizer_betas: tuple[float, float]
    optimizer_weight_decay: float
    optimizer_eps: float
    compile_mode: str | None = None
    source_token_budgets: tuple[tuple[str, int], ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.data_contract_sha256, str) or not SHA256_PATTERN.fullmatch(
            self.data_contract_sha256
        ):
            raise ValueError("data_contract_sha256 must be a lowercase SHA-256 hex digest")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError("seed must be an integer")
        if (
            not isinstance(self.batch_size, int)
            or isinstance(self.batch_size, bool)
            or self.batch_size <= 0
        ):
            raise ValueError("batch_size must be a positive integer")
        if not isinstance(self.optimizer_name, str) or not self.optimizer_name:
            raise ValueError("optimizer_name must be a non-empty string")
        if (
            not isinstance(self.optimizer_betas, tuple)
            or len(self.optimizer_betas) != 2
            or any(
                not isinstance(beta, Real)
                or isinstance(beta, bool)
                or not math.isfinite(float(beta))
                or not 0 <= beta < 1
                for beta in self.optimizer_betas
            )
        ):
            raise ValueError("optimizer_betas must contain two finite values in [0, 1)")
        if (
            not isinstance(self.optimizer_weight_decay, Real)
            or isinstance(self.optimizer_weight_decay, bool)
            or not math.isfinite(float(self.optimizer_weight_decay))
            or self.optimizer_weight_decay < 0
        ):
            raise ValueError("optimizer_weight_decay must be finite and non-negative")
        if (
            not isinstance(self.optimizer_eps, Real)
            or isinstance(self.optimizer_eps, bool)
            or not math.isfinite(float(self.optimizer_eps))
            or self.optimizer_eps <= 0
        ):
            raise ValueError("optimizer_eps must be finite and positive")
        if self.compile_mode not in (None, "default", "reduce-overhead", "max-autotune"):
            raise ValueError(
                "compile_mode must be one of None, 'default', 'reduce-overhead', or 'max-autotune'"
            )
        if self.source_token_budgets is not None:
            if not isinstance(self.source_token_budgets, tuple) or not self.source_token_budgets:
                raise ValueError("source_token_budgets must be a non-empty tuple or None")
            source_names: set[str] = set()
            for budget in self.source_token_budgets:
                if not isinstance(budget, tuple) or len(budget) != 2:
                    raise ValueError("source_token_budgets entries must be (name, tokens) tuples")
                source_name, token_count = budget
                if not isinstance(source_name, str) or not source_name:
                    raise ValueError("source_token_budgets names must be non-empty strings")
                if source_name in source_names:
                    raise ValueError("source_token_budgets names must be unique")
                source_names.add(source_name)
                if (
                    not isinstance(token_count, int)
                    or isinstance(token_count, bool)
                    or token_count <= 0
                ):
                    raise ValueError("source_token_budgets values must be positive integers")


def _validate_optional_string(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string or None")
    return value


def _validate_source_tokens_seen(
    value: object,
    tokens_seen: int,
) -> dict[str, int] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or not value:
        raise ValueError("source_tokens_seen must be a non-empty mapping or None")
    normalized: dict[str, int] = {}
    for source_name, token_count in value.items():
        if not isinstance(source_name, str) or not source_name:
            raise ValueError("source_tokens_seen keys must be non-empty strings")
        if not isinstance(token_count, int) or isinstance(token_count, bool) or token_count < 0:
            raise ValueError("source_tokens_seen values must be non-negative integers")
        normalized[source_name] = token_count
    if sum(normalized.values()) != tokens_seen:
        raise ValueError("source_tokens_seen values must sum to tokens_seen")
    return normalized


def _validate_optional_validation_loss(value: object) -> float | None:
    if value is None:
        return None
    if (
        not isinstance(value, Real)
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or value < 0
    ):
        raise ValueError("validation_loss must be finite and non-negative or None")
    return float(value)


def save_checkpoint(
    path: str | Path,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    step: int,
    training_config: TrainingConfig,
    wandb_run_id: str | None = None,
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
    keep_last_n: int | None = None,
    run_config: TrainingRunConfig | None = None,
    data_position: int = 0,
    tokens_seen: int = 0,
    source_tokens_seen: Mapping[str, int] | None = None,
    validation_loss: float | None = None,
    best_validation_loss: float | None = None,
) -> Path:
    """Atomically save the state required to resume training."""
    if not isinstance(step, int) or isinstance(step, bool) or step < 0:
        raise ValueError(f"step must be a non-negative integer, got {step!r}")
    if step > training_config.max_steps:
        raise ValueError(
            "step cannot exceed training_config.max_steps, "
            f"got {step} > {training_config.max_steps}"
        )
    _validate_optional_string(wandb_run_id, "wandb_run_id")
    _validate_optional_string(wandb_project, "wandb_project")
    _validate_optional_string(wandb_entity, "wandb_entity")
    if keep_last_n is not None and (
        not isinstance(keep_last_n, int) or isinstance(keep_last_n, bool) or keep_last_n <= 0
    ):
        raise ValueError("keep_last_n must be a positive integer or None")
    if not isinstance(data_position, int) or isinstance(data_position, bool) or data_position < 0:
        raise ValueError("data_position must be a non-negative integer")
    if not isinstance(tokens_seen, int) or isinstance(tokens_seen, bool) or tokens_seen < 0:
        raise ValueError("tokens_seen must be a non-negative integer")
    normalized_source_tokens_seen = _validate_source_tokens_seen(
        source_tokens_seen,
        tokens_seen,
    )
    normalized_validation_loss = _validate_optional_validation_loss(validation_loss)
    normalized_best_validation_loss = _validate_optional_validation_loss(best_validation_loss)
    if (
        normalized_validation_loss is not None
        and normalized_best_validation_loss is not None
        and normalized_best_validation_loss > normalized_validation_loss
    ):
        raise ValueError("best_validation_loss cannot exceed validation_loss")

    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = checkpoint_path.with_name(f"{checkpoint_path.name}.tmp")
    rng_snapshot = capture_torch_rng_state()
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "step": step,
        "torch_rng_state": rng_snapshot.cpu_state,
        "cuda_rng_states": rng_snapshot.cuda_states,
        "mps_rng_state": rng_snapshot.mps_state,
        "model_config": asdict(model.config),
        "training_config": asdict(training_config),
        "wandb_run_id": wandb_run_id,
        "wandb_project": wandb_project,
        "wandb_entity": wandb_entity,
        "run_config": None if run_config is None else asdict(run_config),
        "data_position": data_position,
        "tokens_seen": tokens_seen,
        "source_tokens_seen": normalized_source_tokens_seen,
        # Best-checkpoint selection must survive process restarts. Storing the
        # metric beside the exact weights makes that decision auditable.
        "validation_loss": normalized_validation_loss,
        "best_validation_loss": normalized_best_validation_loss,
    }

    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, checkpoint_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    if keep_last_n is not None:
        for obsolete_path in _checkpoint_paths(checkpoint_path.parent)[:-keep_last_n]:
            obsolete_path.unlink()

    return checkpoint_path


def _checkpoint_paths(directory: Path) -> list[Path]:
    checkpoints: list[tuple[int, Path]] = []
    if not directory.is_dir():
        return []
    for path in directory.iterdir():
        match = CHECKPOINT_FILENAME_PATTERN.fullmatch(path.name)
        if match is not None and path.is_file():
            checkpoints.append((int(match.group(1)), path))
    return [path for _, path in sorted(checkpoints)]


def restore_checkpoint(
    path: str | Path,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    training_config: TrainingConfig,
    map_location: torch.device | str = "cpu",
    run_config: TrainingRunConfig | None = None,
) -> RestoredCheckpoint:
    """Restore model, optimizer, random state, and run identity."""
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
    format_version = payload.get("format_version")
    if not isinstance(format_version, int) or isinstance(format_version, bool):
        raise ValueError(f"checkpoint format_version must be an integer, got {format_version!r}")
    if format_version not in SUPPORTED_CHECKPOINT_FORMAT_VERSIONS:
        raise ValueError(f"Unsupported checkpoint format_version: {format_version!r}")

    step = payload.get("step")
    if not isinstance(step, int) or isinstance(step, bool) or step < 0:
        raise ValueError(f"checkpoint contains an invalid step: {step!r}")
    filename_match = CHECKPOINT_FILENAME_PATTERN.fullmatch(checkpoint_path.name)
    if filename_match is not None and int(filename_match.group(1)) != step:
        raise ValueError(
            f"checkpoint filename step {int(filename_match.group(1))} "
            f"does not match payload step {step}"
        )

    model_state = payload.get("model_state")
    optimizer_state = payload.get("optimizer_state")
    torch_rng_state = payload.get("torch_rng_state")
    model_config = payload.get("model_config")
    saved_training_config = payload.get("training_config")
    wandb_run_id = payload.get("wandb_run_id")
    wandb_project = payload.get("wandb_project")
    wandb_entity = payload.get("wandb_entity")
    saved_run_config = payload.get("run_config")
    data_position = payload.get("data_position", 0)
    tokens_seen = payload.get("tokens_seen", 0)
    source_tokens_seen = payload.get("source_tokens_seen")
    validation_loss = payload.get("validation_loss")
    best_validation_loss = payload.get("best_validation_loss")
    cuda_rng_states = payload.get("cuda_rng_states")
    mps_rng_state = payload.get("mps_rng_state")
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
    saved_training_config = dict(saved_training_config)
    if format_version == 1:
        saved_training_config.setdefault("precision", "fp32")
    if format_version <= 3:
        saved_training_config.setdefault("learning_rate_schedule", "cosine")
        saved_training_config.setdefault("decay_start_step", None)
    wandb_run_id = _validate_optional_string(wandb_run_id, "checkpoint wandb_run_id")
    wandb_project = _validate_optional_string(wandb_project, "checkpoint wandb_project")
    wandb_entity = _validate_optional_string(wandb_entity, "checkpoint wandb_entity")
    if saved_run_config is not None and not isinstance(saved_run_config, dict):
        raise ValueError("checkpoint contains an invalid run_config")
    if cuda_rng_states is not None and (
        not isinstance(cuda_rng_states, list)
        or not all(isinstance(state, torch.Tensor) for state in cuda_rng_states)
    ):
        raise ValueError("checkpoint contains invalid cuda_rng_states")
    if mps_rng_state is not None and not isinstance(mps_rng_state, torch.Tensor):
        raise ValueError("checkpoint contains an invalid mps_rng_state")
    if not isinstance(data_position, int) or isinstance(data_position, bool) or data_position < 0:
        raise ValueError("checkpoint contains an invalid data_position")
    if not isinstance(tokens_seen, int) or isinstance(tokens_seen, bool) or tokens_seen < 0:
        raise ValueError("checkpoint contains an invalid tokens_seen")
    source_tokens_seen = _validate_source_tokens_seen(source_tokens_seen, tokens_seen)
    validation_loss = _validate_optional_validation_loss(validation_loss)
    best_validation_loss = _validate_optional_validation_loss(best_validation_loss)
    if (
        validation_loss is not None
        and best_validation_loss is not None
        and best_validation_loss > validation_loss
    ):
        raise ValueError("checkpoint best_validation_loss cannot exceed validation_loss")
    if model_config != asdict(model.config):
        raise ValueError("checkpoint model_config does not match the current model configuration")
    if saved_training_config != asdict(training_config):
        raise ValueError(
            "checkpoint training_config does not match the current training configuration"
        )
    expected_run_config = None if run_config is None else asdict(run_config)
    if expected_run_config is not None:
        if format_version == 1 and saved_run_config is None:
            warnings.warn(
                "Legacy checkpoint has no run_config; cannot validate data, seed, batch, "
                "or optimizer settings",
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            normalized_run_config = None if saved_run_config is None else dict(saved_run_config)
            if normalized_run_config is not None and format_version <= 3:
                legacy_digest = normalized_run_config.pop("manifest_sha256", None)
                normalized_run_config["data_contract_sha256"] = legacy_digest
                normalized_run_config["seed"] = expected_run_config["seed"]
                warnings.warn(
                    "Legacy checkpoint has no seed in run_config; the seed cannot be validated",
                    RuntimeWarning,
                    stacklevel=2,
                )
            if normalized_run_config is not None and format_version <= 4:
                normalized_run_config.setdefault("compile_mode", None)
            if normalized_run_config is not None and format_version <= 4:
                normalized_run_config["source_token_budgets"] = expected_run_config[
                    "source_token_budgets"
                ]
                if expected_run_config["source_token_budgets"] is not None:
                    warnings.warn(
                        "Legacy checkpoint has no source token budgets; the configured "
                        "mixture cannot be validated independently of its data contract digest",
                        RuntimeWarning,
                        stacklevel=2,
                    )
            if normalized_run_config != expected_run_config:
                raise ValueError(
                    "checkpoint run_config does not match the current run configuration"
                )
    if step > training_config.max_steps:
        raise ValueError(
            "checkpoint step cannot exceed training_config.max_steps, "
            f"got {step} > {training_config.max_steps}"
        )

    model.load_state_dict(cast(dict[str, torch.Tensor], model_state))
    optimizer.load_state_dict(cast(dict[str, object], optimizer_state))
    restore_torch_rng_state(
        TorchRngSnapshot(
            cpu_state=torch_rng_state,
            cuda_states=cast(list[torch.Tensor] | None, cuda_rng_states),
            mps_state=mps_rng_state,
        )
    )

    return RestoredCheckpoint(
        path=checkpoint_path,
        step=step,
        wandb_run_id=wandb_run_id,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        data_position=data_position,
        tokens_seen=tokens_seen,
        source_tokens_seen=source_tokens_seen,
        validation_loss=validation_loss,
        best_validation_loss=best_validation_loss,
    )


def load_checkpoint(
    path: str | Path,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    training_config: TrainingConfig,
    map_location: torch.device | str = "cpu",
    run_config: TrainingRunConfig | None = None,
) -> int:
    """Restore a checkpoint and return its completed optimizer step."""
    return restore_checkpoint(
        path=path,
        model=model,
        optimizer=optimizer,
        training_config=training_config,
        map_location=map_location,
        run_config=run_config,
    ).step


def restore_latest_checkpoint(
    directory: str | Path,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    training_config: TrainingConfig,
    map_location: torch.device | str = "cpu",
    run_config: TrainingRunConfig | None = None,
) -> RestoredCheckpoint:
    """Restore the newest readable compatible checkpoint in a directory."""
    checkpoint_paths = list(reversed(_checkpoint_paths(Path(directory))))
    if not checkpoint_paths:
        raise FileNotFoundError(f"No step checkpoints found in {Path(directory)}")

    last_error: Exception | None = None
    for checkpoint_path in checkpoint_paths:
        try:
            return restore_checkpoint(
                path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                training_config=training_config,
                map_location=map_location,
                run_config=run_config,
            )
        except Exception as error:
            last_error = error
            warnings.warn(
                f"Skipping unreadable checkpoint {checkpoint_path}: {error}",
                RuntimeWarning,
                stacklevel=2,
            )

    raise RuntimeError(f"No valid step checkpoints found in {Path(directory)}") from last_error
