from pathlib import Path

import pytest
import torch

from model.config import ModelConfig
from model.gpt import GPT
from training.checkpoint import TrainingRunConfig, restore_latest_checkpoint
from training.observers import CheckpointWriter
from training.trainer import TrainingConfig


def _build_writer(
    directory: Path,
    *,
    best_validation_loss: float | None = None,
) -> CheckpointWriter:
    model = GPT(
        ModelConfig(
            vocab_size=8,
            d_model=8,
            n_layers=1,
            n_heads=2,
            max_seq_len=4,
            dropout=0.0,
        )
    )
    optimizer = torch.optim.AdamW(model.parameters())
    return CheckpointWriter(
        directory=directory,
        model=model,
        optimizer=optimizer,
        training_config=TrainingConfig(max_steps=4),
        run_config=TrainingRunConfig(
            data_contract_sha256="0" * 64,
            seed=42,
            batch_size=2,
            optimizer_name="AdamW",
            optimizer_betas=(0.9, 0.999),
            optimizer_weight_decay=0.01,
            optimizer_eps=1e-8,
        ),
        sequences_per_step=2,
        wandb_run_id=None,
        wandb_project=None,
        wandb_entity=None,
        keep_last_n=1,
        best_validation_loss=best_validation_loss,
    )


def test_best_checkpoint_survives_worse_evaluations_retention_and_resume(
    tmp_path: Path,
) -> None:
    writer = _build_writer(tmp_path)

    assert writer.save_best_if_improved(
        step=1,
        tokens_seen=8,
        source_tokens_seen=None,
        validation_loss=1.0,
    )
    writer.save_step(
        step=1,
        tokens_seen=8,
        source_tokens_seen=None,
        validation_loss=1.0,
    )
    assert not writer.save_best_if_improved(
        step=2,
        tokens_seen=16,
        source_tokens_seen=None,
        validation_loss=1.5,
    )
    writer.save_step(
        step=2,
        tokens_seen=16,
        source_tokens_seen=None,
        validation_loss=1.5,
    )

    best_path = tmp_path / "best_validation.pt"
    best_checkpoint = torch.load(best_path, weights_only=True)
    assert best_checkpoint["step"] == 1
    assert best_checkpoint["validation_loss"] == pytest.approx(1.0)
    assert [path.name for path in tmp_path.glob("step_*.pt")] == ["step_000002.pt"]

    resumed_writer = _build_writer(tmp_path)
    restored = restore_latest_checkpoint(
        directory=tmp_path,
        model=resumed_writer.model,
        optimizer=resumed_writer.optimizer,
        training_config=resumed_writer.training_config,
        run_config=resumed_writer.run_config,
    )
    resumed_writer.best_validation_loss = restored.best_validation_loss

    assert restored.best_validation_loss == pytest.approx(1.0)
    assert not resumed_writer.save_best_if_improved(
        step=3,
        tokens_seen=24,
        source_tokens_seen=None,
        validation_loss=1.2,
    )
    resumed_writer.save_step(
        step=3,
        tokens_seen=24,
        source_tokens_seen=None,
        validation_loss=1.2,
    )

    best_checkpoint = torch.load(best_path, weights_only=True)
    assert best_checkpoint["step"] == 1
    assert best_checkpoint["validation_loss"] == pytest.approx(1.0)
    assert [path.name for path in tmp_path.glob("step_*.pt")] == ["step_000003.pt"]


def test_failed_best_checkpoint_save_does_not_advance_selection_threshold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writer = _build_writer(tmp_path, best_validation_loss=1.0)

    def fail_save_checkpoint(**_: object) -> Path:
        raise OSError("simulated checkpoint failure")

    monkeypatch.setattr("training.observers.save_checkpoint", fail_save_checkpoint)

    with pytest.raises(OSError, match="simulated checkpoint failure"):
        writer.save_best_if_improved(
            step=2,
            tokens_seen=16,
            source_tokens_seen=None,
            validation_loss=0.8,
        )

    assert writer.best_validation_loss == pytest.approx(1.0)
