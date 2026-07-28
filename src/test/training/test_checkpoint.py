from pathlib import Path

import pytest
import torch

from model.config import ModelConfig
from model.gpt import GPT
from training.checkpoint import load_checkpoint, save_checkpoint
from training.trainer import TrainingConfig, train_step


def _tiny_model_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=8,
        d_model=16,
        n_layers=2,
        n_heads=4,
        max_seq_len=6,
        dropout=0.0,
    )


def test_checkpoint_restores_training_state_and_can_continue(tmp_path: Path) -> None:
    torch.manual_seed(42)
    model_config = _tiny_model_config()
    training_config = TrainingConfig(
        max_steps=4,
        grad_accum_steps=1,
        max_learning_rate=1e-2,
        min_learning_rate=1e-3,
        max_grad_norm=1.0,
    )
    model = GPT(model_config)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.max_learning_rate,
        weight_decay=0.0,
    )
    batch = (
        torch.tensor([[0, 1, 2, 3, 4, 5]]),
        torch.tensor([[1, 2, 3, 4, 5, 6]]),
    )
    train_step(
        model=model,
        optimizer=optimizer,
        microbatches=[batch],
        max_grad_norm=training_config.max_grad_norm,
        device="cpu",
    )
    checkpoint_path = tmp_path / "nested" / "step_000001.pt"

    saved_path = save_checkpoint(
        path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        step=1,
        training_config=training_config,
    )
    expected_random_values = torch.rand(4)

    restored_model = GPT(model_config)
    restored_optimizer = torch.optim.AdamW(
        restored_model.parameters(),
        lr=training_config.max_learning_rate,
        weight_decay=0.0,
    )
    loaded_step = load_checkpoint(
        path=checkpoint_path,
        model=restored_model,
        optimizer=restored_optimizer,
        training_config=training_config,
        map_location="cpu",
    )
    restored_random_values = torch.rand(4)

    assert saved_path == checkpoint_path
    assert loaded_step == 1
    torch.testing.assert_close(restored_random_values, expected_random_values)
    for expected, restored in zip(
        model.state_dict().values(),
        restored_model.state_dict().values(),
        strict=True,
    ):
        torch.testing.assert_close(restored, expected, rtol=0.0, atol=0.0)
    assert restored_optimizer.param_groups[0]["lr"] == optimizer.param_groups[0]["lr"]

    train_step(
        model=model,
        optimizer=optimizer,
        microbatches=[batch],
        max_grad_norm=training_config.max_grad_norm,
        device="cpu",
    )
    train_step(
        model=restored_model,
        optimizer=restored_optimizer,
        microbatches=[batch],
        max_grad_norm=training_config.max_grad_norm,
        device="cpu",
    )

    for expected, restored in zip(
        model.state_dict().values(),
        restored_model.state_dict().values(),
        strict=True,
    ):
        torch.testing.assert_close(restored, expected, rtol=0.0, atol=0.0)


def test_checkpoint_rejects_a_different_model_configuration(tmp_path: Path) -> None:
    training_config = TrainingConfig(max_steps=1)
    model = GPT(_tiny_model_config())
    optimizer = torch.optim.AdamW(model.parameters())
    checkpoint_path = save_checkpoint(
        path=tmp_path / "checkpoint.pt",
        model=model,
        optimizer=optimizer,
        step=0,
        training_config=training_config,
    )
    incompatible_config = ModelConfig(
        vocab_size=8,
        d_model=16,
        n_layers=2,
        n_heads=4,
        max_seq_len=6,
        dropout=0.0,
        use_sdpa=False,
    )
    incompatible_model = GPT(incompatible_config)
    incompatible_optimizer = torch.optim.AdamW(incompatible_model.parameters())

    with pytest.raises(
        ValueError,
        match="checkpoint model_config does not match the current model configuration",
    ):
        load_checkpoint(
            path=checkpoint_path,
            model=incompatible_model,
            optimizer=incompatible_optimizer,
            training_config=training_config,
        )


def test_checkpoint_rejects_a_different_training_configuration(
    tmp_path: Path,
) -> None:
    model = GPT(_tiny_model_config())
    optimizer = torch.optim.AdamW(model.parameters())
    saved_training_config = TrainingConfig(max_steps=4, grad_accum_steps=2)
    checkpoint_path = save_checkpoint(
        path=tmp_path / "checkpoint.pt",
        model=model,
        optimizer=optimizer,
        step=2,
        training_config=saved_training_config,
    )
    different_training_config = TrainingConfig(max_steps=4, grad_accum_steps=1)

    with pytest.raises(
        ValueError,
        match=("checkpoint training_config does not match the current training configuration"),
    ):
        load_checkpoint(
            path=checkpoint_path,
            model=model,
            optimizer=optimizer,
            training_config=different_training_config,
        )
