from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from evaluation.harness_adapter import load_evaluation_checkpoint
from model.config import ModelConfig
from model.gpt import GPT
from training.checkpoint import save_checkpoint
from training.trainer import TrainingConfig


def test_evaluation_checkpoint_reconstructs_model_from_saved_configuration(
    tmp_path: Path,
) -> None:
    torch.manual_seed(42)
    model = GPT(
        ModelConfig(
            vocab_size=32,
            d_model=16,
            n_layers=2,
            n_heads=4,
            max_seq_len=8,
        )
    )
    optimizer = torch.optim.AdamW(model.parameters())
    checkpoint_path = tmp_path / "step_000003.pt"
    save_checkpoint(
        path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        step=3,
        training_config=TrainingConfig(max_steps=4),
    )

    loaded = load_evaluation_checkpoint(checkpoint_path, device="cpu")

    assert loaded.step == 3
    assert asdict(loaded.model.config) == asdict(model.config)
    assert not loaded.model.training
    for name, expected in model.state_dict().items():
        torch.testing.assert_close(loaded.model.state_dict()[name], expected)


def test_evaluation_checkpoint_rejects_an_unsupported_format(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "format_version": 999,
            "step": 0,
            "model_config": asdict(
                ModelConfig(
                    vocab_size=8,
                    d_model=8,
                    n_layers=1,
                    n_heads=2,
                    max_seq_len=4,
                )
            ),
            "model_state": {},
        },
        checkpoint_path,
    )

    with pytest.raises(ValueError, match="Unsupported checkpoint format_version"):
        load_evaluation_checkpoint(checkpoint_path)
