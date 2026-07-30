from pathlib import Path

import pytest
import torch

from model.config import ModelConfig
from model.gpt import GPT
from training.checkpoint import (
    TrainingRunConfig,
    load_checkpoint,
    restore_checkpoint,
    restore_latest_checkpoint,
    save_checkpoint,
)
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


def test_checkpoint_restores_the_wandb_run_identity(tmp_path: Path) -> None:
    model = GPT(_tiny_model_config())
    optimizer = torch.optim.AdamW(model.parameters())
    training_config = TrainingConfig(max_steps=1)
    checkpoint_path = save_checkpoint(
        path=tmp_path / "step_000000.pt",
        model=model,
        optimizer=optimizer,
        step=0,
        training_config=training_config,
        wandb_run_id="wandb-run-123",
        wandb_project="llm-from-scratch",
        wandb_entity="research-team",
        data_position=64,
        tokens_seen=384,
    )

    restored = restore_checkpoint(
        path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        training_config=training_config,
    )

    assert restored.path == checkpoint_path
    assert restored.step == 0
    assert restored.wandb_run_id == "wandb-run-123"
    assert restored.wandb_project == "llm-from-scratch"
    assert restored.wandb_entity == "research-team"
    assert restored.data_position == 64
    assert restored.tokens_seen == 384


def test_checkpoint_retention_keeps_only_the_newest_files(tmp_path: Path) -> None:
    model = GPT(_tiny_model_config())
    optimizer = torch.optim.AdamW(model.parameters())
    training_config = TrainingConfig(max_steps=4)

    for step in range(1, 5):
        save_checkpoint(
            path=tmp_path / f"step_{step:06d}.pt",
            model=model,
            optimizer=optimizer,
            step=step,
            training_config=training_config,
            keep_last_n=2,
        )

    assert sorted(path.name for path in tmp_path.glob("step_*.pt")) == [
        "step_000003.pt",
        "step_000004.pt",
    ]


def test_latest_checkpoint_falls_back_when_the_newest_file_is_corrupt(
    tmp_path: Path,
) -> None:
    model = GPT(_tiny_model_config())
    optimizer = torch.optim.AdamW(model.parameters())
    training_config = TrainingConfig(max_steps=2)
    save_checkpoint(
        path=tmp_path / "step_000001.pt",
        model=model,
        optimizer=optimizer,
        step=1,
        training_config=training_config,
        wandb_run_id="wandb-run-123",
    )
    (tmp_path / "step_000002.pt").write_bytes(b"not a torch checkpoint")

    with pytest.warns(RuntimeWarning, match="Skipping unreadable checkpoint"):
        restored = restore_latest_checkpoint(
            directory=tmp_path,
            model=model,
            optimizer=optimizer,
            training_config=training_config,
        )

    assert restored.path == tmp_path / "step_000001.pt"
    assert restored.step == 1
    assert restored.wandb_run_id == "wandb-run-123"


def test_latest_checkpoint_falls_back_when_the_newest_config_is_incompatible(
    tmp_path: Path,
) -> None:
    model = GPT(_tiny_model_config())
    optimizer = torch.optim.AdamW(model.parameters())
    requested_config = TrainingConfig(max_steps=2, grad_accum_steps=1)
    save_checkpoint(
        path=tmp_path / "step_000001.pt",
        model=model,
        optimizer=optimizer,
        step=1,
        training_config=requested_config,
    )
    save_checkpoint(
        path=tmp_path / "step_000002.pt",
        model=model,
        optimizer=optimizer,
        step=2,
        training_config=TrainingConfig(max_steps=2, grad_accum_steps=2),
    )

    with pytest.warns(RuntimeWarning, match="Skipping unreadable checkpoint"):
        restored = restore_latest_checkpoint(
            directory=tmp_path,
            model=model,
            optimizer=optimizer,
            training_config=requested_config,
        )

    assert restored.path == tmp_path / "step_000001.pt"
    assert restored.step == 1


def test_latest_checkpoint_falls_back_when_filename_step_disagrees_with_payload(
    tmp_path: Path,
) -> None:
    model = GPT(_tiny_model_config())
    optimizer = torch.optim.AdamW(model.parameters())
    training_config = TrainingConfig(max_steps=2)
    save_checkpoint(
        path=tmp_path / "step_000001.pt",
        model=model,
        optimizer=optimizer,
        step=1,
        training_config=training_config,
    )
    newest_path = save_checkpoint(
        path=tmp_path / "step_000002.pt",
        model=model,
        optimizer=optimizer,
        step=2,
        training_config=training_config,
    )
    payload = torch.load(newest_path, weights_only=True)
    payload["step"] = 1
    torch.save(payload, newest_path)

    with pytest.warns(RuntimeWarning, match="filename step"):
        restored = restore_latest_checkpoint(
            directory=tmp_path,
            model=model,
            optimizer=optimizer,
            training_config=training_config,
        )

    assert restored.path == tmp_path / "step_000001.pt"
    assert restored.step == 1


def test_checkpoint_validates_the_training_run_configuration(tmp_path: Path) -> None:
    model = GPT(_tiny_model_config())
    optimizer = torch.optim.AdamW(model.parameters())
    training_config = TrainingConfig(max_steps=1)
    saved_run_config = TrainingRunConfig(
        data_contract_sha256="a" * 64,
        seed=42,
        batch_size=2,
        optimizer_name="AdamW",
        optimizer_betas=(0.9, 0.95),
        optimizer_weight_decay=0.1,
        optimizer_eps=1e-8,
        source_token_budgets=(("web", 12), ("math", 4)),
    )
    checkpoint_path = save_checkpoint(
        path=tmp_path / "step_000000.pt",
        model=model,
        optimizer=optimizer,
        step=0,
        training_config=training_config,
        run_config=saved_run_config,
    )
    different_run_config = TrainingRunConfig(
        data_contract_sha256="a" * 64,
        seed=43,
        batch_size=2,
        optimizer_name="AdamW",
        optimizer_betas=(0.9, 0.95),
        optimizer_weight_decay=0.1,
        optimizer_eps=1e-8,
        source_token_budgets=(("web", 12), ("math", 4)),
    )

    with pytest.raises(ValueError, match="checkpoint run_config does not match"):
        restore_checkpoint(
            path=checkpoint_path,
            model=model,
            optimizer=optimizer,
            training_config=training_config,
            run_config=different_run_config,
        )

    different_run_config = TrainingRunConfig(
        data_contract_sha256="a" * 64,
        seed=42,
        batch_size=2,
        optimizer_name="AdamW",
        optimizer_betas=(0.9, 0.95),
        optimizer_weight_decay=0.1,
        optimizer_eps=1e-8,
        compile_mode="default",
        source_token_budgets=(("web", 12), ("math", 4)),
    )

    with pytest.raises(ValueError, match="checkpoint run_config does not match"):
        restore_checkpoint(
            path=checkpoint_path,
            model=model,
            optimizer=optimizer,
            training_config=training_config,
            run_config=different_run_config,
        )

    different_run_config = TrainingRunConfig(
        data_contract_sha256="a" * 64,
        seed=42,
        batch_size=4,
        optimizer_name="AdamW",
        optimizer_betas=(0.9, 0.95),
        optimizer_weight_decay=0.1,
        optimizer_eps=1e-8,
        source_token_budgets=(("web", 12), ("math", 4)),
    )

    with pytest.raises(ValueError, match="checkpoint run_config does not match"):
        restore_checkpoint(
            path=checkpoint_path,
            model=model,
            optimizer=optimizer,
            training_config=training_config,
            run_config=different_run_config,
        )


def test_checkpoint_persists_the_configured_recipe_mix(tmp_path: Path) -> None:
    model = GPT(_tiny_model_config())
    optimizer = torch.optim.AdamW(model.parameters())
    training_config = TrainingConfig(max_steps=1)
    run_config = TrainingRunConfig(
        data_contract_sha256="a" * 64,
        seed=42,
        batch_size=2,
        optimizer_name="AdamW",
        optimizer_betas=(0.9, 0.95),
        optimizer_weight_decay=0.1,
        optimizer_eps=1e-8,
        source_token_budgets=(("web", 12), ("math", 4)),
    )

    checkpoint_path = save_checkpoint(
        path=tmp_path / "step_000000.pt",
        model=model,
        optimizer=optimizer,
        step=0,
        training_config=training_config,
        run_config=run_config,
    )

    payload = torch.load(checkpoint_path, weights_only=True)

    assert payload["run_config"]["source_token_budgets"] == (("web", 12), ("math", 4))


def test_checkpoint_preserves_exact_tokens_consumed_per_recipe_source(
    tmp_path: Path,
) -> None:
    model = GPT(_tiny_model_config())
    optimizer = torch.optim.AdamW(model.parameters())
    training_config = TrainingConfig(max_steps=1)
    checkpoint_path = save_checkpoint(
        path=tmp_path / "step_000001.pt",
        model=model,
        optimizer=optimizer,
        step=1,
        training_config=training_config,
        tokens_seen=16,
        source_tokens_seen={"web": 12, "math": 4},
    )

    restored = restore_checkpoint(
        path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        training_config=training_config,
    )

    assert restored.source_tokens_seen == {"web": 12, "math": 4}


def test_checkpoint_v1_without_precision_loads_as_fp32(tmp_path: Path) -> None:
    model = GPT(_tiny_model_config())
    optimizer = torch.optim.AdamW(model.parameters())
    training_config = TrainingConfig(max_steps=1, precision="fp32")
    checkpoint_path = save_checkpoint(
        path=tmp_path / "step_000000.pt",
        model=model,
        optimizer=optimizer,
        step=0,
        training_config=training_config,
    )
    payload = torch.load(checkpoint_path, weights_only=True)
    payload["format_version"] = 1
    payload["training_config"].pop("precision")
    payload.pop("wandb_run_id")
    payload.pop("run_config")
    torch.save(payload, checkpoint_path)

    restored = restore_checkpoint(
        path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        training_config=training_config,
    )

    assert restored.step == 0


def test_checkpoint_v1_without_run_config_loads_with_an_explicit_warning(
    tmp_path: Path,
) -> None:
    model = GPT(_tiny_model_config())
    optimizer = torch.optim.AdamW(model.parameters())
    training_config = TrainingConfig(max_steps=1)
    checkpoint_path = save_checkpoint(
        path=tmp_path / "step_000000.pt",
        model=model,
        optimizer=optimizer,
        step=0,
        training_config=training_config,
    )
    payload = torch.load(checkpoint_path, weights_only=True)
    payload["format_version"] = 1
    payload["training_config"].pop("precision")
    payload.pop("run_config")
    torch.save(payload, checkpoint_path)
    run_config = TrainingRunConfig(
        data_contract_sha256="a" * 64,
        seed=42,
        batch_size=2,
        optimizer_name="AdamW",
        optimizer_betas=(0.9, 0.95),
        optimizer_weight_decay=0.1,
        optimizer_eps=1e-8,
    )

    with pytest.warns(RuntimeWarning, match="cannot validate data, seed, batch"):
        restored = restore_checkpoint(
            path=checkpoint_path,
            model=model,
            optimizer=optimizer,
            training_config=training_config,
            run_config=run_config,
        )

    assert restored.step == 0


def test_legacy_recipe_checkpoint_uses_the_expected_source_budgets_with_a_warning(
    tmp_path: Path,
) -> None:
    model = GPT(_tiny_model_config())
    optimizer = torch.optim.AdamW(model.parameters())
    training_config = TrainingConfig(max_steps=1)
    run_config = TrainingRunConfig(
        data_contract_sha256="a" * 64,
        seed=42,
        batch_size=2,
        optimizer_name="AdamW",
        optimizer_betas=(0.9, 0.95),
        optimizer_weight_decay=0.1,
        optimizer_eps=1e-8,
        source_token_budgets=(("web", 12), ("math", 4)),
    )
    checkpoint_path = save_checkpoint(
        path=tmp_path / "step_000000.pt",
        model=model,
        optimizer=optimizer,
        step=0,
        training_config=training_config,
        run_config=run_config,
    )
    payload = torch.load(checkpoint_path, weights_only=True)
    payload["format_version"] = 4
    payload["run_config"].pop("source_token_budgets")
    torch.save(payload, checkpoint_path)

    with pytest.warns(RuntimeWarning, match="has no source token budgets"):
        restored = restore_checkpoint(
            path=checkpoint_path,
            model=model,
            optimizer=optimizer,
            training_config=training_config,
            run_config=run_config,
        )

    assert restored.step == 0


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("data_contract_sha256", "not-a-digest", "data_contract_sha256"),
        ("seed", True, "seed"),
        ("batch_size", 0, "batch_size"),
        ("optimizer_name", "", "optimizer_name"),
        ("optimizer_betas", (0.9, 1.0), "optimizer_betas"),
        ("optimizer_weight_decay", -0.1, "optimizer_weight_decay"),
        ("optimizer_eps", 0.0, "optimizer_eps"),
        ("compile_mode", "fastest", "compile_mode"),
    ],
)
def test_training_run_config_rejects_invalid_values(
    field: str,
    value: object,
    message: str,
) -> None:
    values: dict[str, object] = {
        "data_contract_sha256": "a" * 64,
        "seed": 42,
        "batch_size": 2,
        "optimizer_name": "AdamW",
        "optimizer_betas": (0.9, 0.95),
        "optimizer_weight_decay": 0.1,
        "optimizer_eps": 1e-8,
    }
    values[field] = value

    with pytest.raises(ValueError, match=message):
        TrainingRunConfig(**values)  # type: ignore[arg-type]


def test_checkpoint_rejects_a_non_integer_format_version(tmp_path: Path) -> None:
    model = GPT(_tiny_model_config())
    optimizer = torch.optim.AdamW(model.parameters())
    training_config = TrainingConfig(max_steps=1)
    checkpoint_path = save_checkpoint(
        path=tmp_path / "checkpoint.pt",
        model=model,
        optimizer=optimizer,
        step=0,
        training_config=training_config,
    )
    payload = torch.load(checkpoint_path, weights_only=True)
    payload["format_version"] = []
    torch.save(payload, checkpoint_path)

    with pytest.raises(ValueError, match="format_version must be an integer"):
        restore_checkpoint(
            path=checkpoint_path,
            model=model,
            optimizer=optimizer,
            training_config=training_config,
        )


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


def test_checkpoint_rejects_a_step_beyond_the_training_horizon(
    tmp_path: Path,
) -> None:
    model = GPT(_tiny_model_config())
    optimizer = torch.optim.AdamW(model.parameters())
    training_config = TrainingConfig(max_steps=1)

    with pytest.raises(
        ValueError,
        match="step cannot exceed training_config.max_steps",
    ):
        save_checkpoint(
            path=tmp_path / "checkpoint.pt",
            model=model,
            optimizer=optimizer,
            step=2,
            training_config=training_config,
        )


def test_checkpoint_rejects_a_stored_step_beyond_the_training_horizon(
    tmp_path: Path,
) -> None:
    model = GPT(_tiny_model_config())
    optimizer = torch.optim.AdamW(model.parameters())
    training_config = TrainingConfig(max_steps=1)
    checkpoint_path = save_checkpoint(
        path=tmp_path / "checkpoint.pt",
        model=model,
        optimizer=optimizer,
        step=1,
        training_config=training_config,
    )
    payload = torch.load(checkpoint_path, weights_only=True)
    payload["step"] = 2
    torch.save(payload, checkpoint_path)

    with pytest.raises(
        ValueError,
        match="checkpoint step cannot exceed training_config.max_steps",
    ):
        load_checkpoint(
            path=checkpoint_path,
            model=model,
            optimizer=optimizer,
            training_config=training_config,
        )


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS is not available",
)
def test_checkpoint_restores_mps_random_state(tmp_path: Path) -> None:
    torch.mps.manual_seed(123)
    model = GPT(_tiny_model_config())
    optimizer = torch.optim.AdamW(model.parameters())
    training_config = TrainingConfig(max_steps=1)
    checkpoint_path = save_checkpoint(
        path=tmp_path / "checkpoint.pt",
        model=model,
        optimizer=optimizer,
        step=0,
        training_config=training_config,
    )
    expected_random_values = torch.rand(4, device="mps").cpu()
    torch.rand(10, device="mps")

    load_checkpoint(
        path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        training_config=training_config,
    )
    restored_random_values = torch.rand(4, device="mps").cpu()

    torch.testing.assert_close(restored_random_values, expected_random_values)
