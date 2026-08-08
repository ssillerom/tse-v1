import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from data.dataset import PretrainingDataset
from data.manifest import FORMAT_VERSION, STORAGE_DTYPE, write_manifest
from model.config import ModelConfig
from model.gpt import GPT
from scripts.train_v1 import main
from training.checkpoint import TrainingRunConfig, save_checkpoint
from training.optimizer import build_adamw_parameter_groups
from training.trainer import TrainingConfig, train


def _write_training_manifest(directory: Path) -> Path:
    train_tokens = np.tile(np.array([100, 101, 102, 103, 104], dtype=np.uint16), 8)
    validation_tokens = np.tile(np.array([100, 101, 102, 103, 104], dtype=np.uint16), 4)
    train_path = directory / "train.bin"
    validation_path = directory / "validation.bin"
    train_tokens.tofile(train_path)
    validation_tokens.tofile(validation_path)
    with train_path.open("rb") as train_file:
        train_sha256 = hashlib.file_digest(train_file, "sha256").hexdigest()
    with validation_path.open("rb") as validation_file:
        validation_sha256 = hashlib.file_digest(validation_file, "sha256").hexdigest()
    manifest_path = directory / "manifest.json"
    write_manifest(
        manifest_path,
        {
            "format_version": FORMAT_VERSION,
            "dataset": {"path": "local/test"},
            "tokenizer": {"encoding": "gpt2", "eot_token": 50_256},
            "storage": {"dtype": STORAGE_DTYPE, "shard_size": 40},
            "counts": {"tokens": 60, "shards": 2},
            "splits": {
                "train": {"tokens": 40, "shards": 1},
                "validation": {"tokens": 20, "shards": 1},
            },
            "shards": [
                {
                    "file": train_path.name,
                    "split": "train",
                    "tokens": 40,
                    "sha256": train_sha256,
                },
                {
                    "file": validation_path.name,
                    "split": "validation",
                    "tokens": 20,
                    "sha256": validation_sha256,
                },
            ],
        },
    )
    return manifest_path


def _write_recipe(directory: Path) -> Path:
    general_directory = directory / "general"
    code_directory = directory / "code"
    general_directory.mkdir()
    code_directory.mkdir()
    general_manifest = _write_training_manifest(general_directory)
    code_manifest = _write_training_manifest(code_directory)
    recipe_path = directory / "recipe.json"
    recipe_path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "name": "tiny-mixed-run",
                "sources": [
                    {
                        "name": "fineweb_edu",
                        "manifest": str(general_manifest.relative_to(directory)),
                    },
                    {
                        "name": "stack_edu",
                        "manifest": str(code_manifest.relative_to(directory)),
                    },
                ],
                "phases": [
                    {
                        "name": "stable",
                        "tokens": 24,
                        "source_tokens": {"fineweb_edu": 16, "stack_edu": 8},
                    },
                    {
                        "name": "decay",
                        "tokens": 8,
                        "source_tokens": {"fineweb_edu": 4, "stack_edu": 4},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return recipe_path


def test_train_v1_runs_evaluation_generation_and_checkpointing_offline(
    tmp_path: Path,
) -> None:
    manifest_path = _write_training_manifest(tmp_path)
    checkpoint_dir = tmp_path / "checkpoints"

    exit_code = main(
        [
            "--manifest",
            str(manifest_path),
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--device",
            "cpu",
            "--precision",
            "bf16",
            "--d-model",
            "8",
            "--n-layers",
            "1",
            "--n-heads",
            "2",
            "--n-kv-heads",
            "1",
            "--seq-len",
            "4",
            "--batch-size",
            "2",
            "--max-steps",
            "1",
            "--warmup-steps",
            "0",
            "--eval-interval",
            "1",
            "--eval-batches",
            "1",
            "--sample-interval",
            "1",
            "--max-new-tokens",
            "1",
            "--checkpoint-interval",
            "2",
            "--wandb-mode",
            "disabled",
        ]
    )

    assert exit_code == 0
    checkpoint_path = checkpoint_dir / "step_000001.pt"
    assert checkpoint_path.is_file()
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    assert checkpoint["model_config"]["n_kv_heads"] == 1
    assert checkpoint["model_state"]["blocks.0.attention.W_k.weight"].shape == (4, 8)
    assert checkpoint["training_config"]["precision"] == "bf16"
    best_checkpoint = torch.load(checkpoint_dir / "best_validation.pt", weights_only=True)
    assert best_checkpoint["validation_loss"] == pytest.approx(
        best_checkpoint["best_validation_loss"]
    )
    assert checkpoint["best_validation_loss"] == pytest.approx(
        best_checkpoint["best_validation_loss"]
    )


def test_train_v1_rejects_torch_compile_without_cuda(tmp_path: Path) -> None:
    manifest_path = _write_training_manifest(tmp_path)

    with pytest.raises(ValueError, match="torch.compile.*CUDA"):
        main(
            [
                "--manifest",
                str(manifest_path),
                "--device",
                "cpu",
                "--compile",
                "--wandb-mode",
                "disabled",
            ]
        )


def test_train_v1_runs_a_complete_mixed_recipe_with_wsd(tmp_path: Path) -> None:
    recipe_path = _write_recipe(tmp_path)
    uninterrupted_dir = tmp_path / "uninterrupted"
    resumed_dir = tmp_path / "resumed"
    common_arguments = [
        "--recipe",
        str(recipe_path),
        "--device",
        "cpu",
        "--precision",
        "fp32",
        "--d-model",
        "8",
        "--n-layers",
        "1",
        "--n-heads",
        "2",
        "--seq-len",
        "4",
        "--dropout",
        "0.2",
        "--batch-size",
        "2",
        "--grad-accum-steps",
        "1",
        "--warmup-steps",
        "1",
        "--lr-schedule",
        "wsd",
        "--eval-interval",
        "2",
        "--eval-batches",
        "1",
        "--sample-interval",
        "4",
        "--max-new-tokens",
        "1",
        "--checkpoint-interval",
        "2",
        "--wandb-mode",
        "disabled",
    ]

    assert (
        main(
            common_arguments
            + [
                "--checkpoint-dir",
                str(uninterrupted_dir),
            ]
        )
        == 0
    )
    assert (
        main(
            common_arguments
            + [
                "--checkpoint-dir",
                str(resumed_dir),
                "--stop-after-step",
                "2",
            ]
        )
        == 0
    )
    assert (
        main(
            common_arguments
            + [
                "--checkpoint-dir",
                str(resumed_dir),
                "--resume",
                "latest",
            ]
        )
        == 0
    )

    checkpoint = torch.load(resumed_dir / "step_000004.pt", weights_only=True)
    uninterrupted_checkpoint = torch.load(
        uninterrupted_dir / "step_000004.pt",
        weights_only=True,
    )
    assert checkpoint["step"] == 4
    assert checkpoint["data_position"] == 8
    assert checkpoint["tokens_seen"] == 32
    assert checkpoint["source_tokens_seen"] == {
        "fineweb_edu": 20,
        "stack_edu": 12,
    }
    assert checkpoint["training_config"]["learning_rate_schedule"] == "wsd"
    assert checkpoint["training_config"]["decay_start_step"] == 3
    assert checkpoint["training_config"]["min_learning_rate"] == 0.0
    for name, expected_parameter in uninterrupted_checkpoint["model_state"].items():
        torch.testing.assert_close(
            checkpoint["model_state"][name],
            expected_parameter,
            rtol=0.0,
            atol=0.0,
        )


def test_train_v1_resumes_from_a_checkpoint_and_preserves_the_wandb_run(
    tmp_path: Path,
) -> None:
    manifest_path = _write_training_manifest(tmp_path)
    checkpoint_dir = tmp_path / "checkpoints"
    torch.manual_seed(42)
    model_config = ModelConfig(
        vocab_size=50_304,
        d_model=8,
        n_layers=1,
        n_heads=2,
        max_seq_len=4,
        dropout=0.2,
        use_sdpa=True,
    )
    training_config = TrainingConfig(
        max_steps=2,
        grad_accum_steps=1,
        warmup_steps=0,
        max_learning_rate=3e-4,
        min_learning_rate=3e-5,
        max_grad_norm=1.0,
        eval_interval=1,
        eval_batches=1,
        precision="fp32",
    )
    model = GPT(model_config)
    optimizer = torch.optim.AdamW(
        build_adamw_parameter_groups(model, weight_decay=0.1),
        lr=training_config.max_learning_rate,
        betas=(0.9, 0.95),
    )
    with manifest_path.open("rb") as manifest_file:
        manifest_sha256 = hashlib.file_digest(manifest_file, "sha256").hexdigest()
    run_config = TrainingRunConfig(
        data_contract_sha256=manifest_sha256,
        seed=42,
        batch_size=2,
        optimizer_name="AdamW",
        optimizer_betas=(0.9, 0.95),
        optimizer_weight_decay=0.1,
        optimizer_eps=1e-8,
    )
    train_loader = DataLoader(
        PretrainingDataset(manifest_path, split="train", seq_len=4),
        batch_size=2,
        shuffle=False,
        drop_last=True,
    )
    validation_loader = DataLoader(
        PretrainingDataset(manifest_path, split="validation", seq_len=4),
        batch_size=2,
        shuffle=False,
        drop_last=False,
    )
    train(
        model=model,
        optimizer=optimizer,
        train_batches=train_loader,
        config=training_config,
        device="cpu",
        validation_batches=validation_loader,
        end_step=1,
    )
    save_checkpoint(
        path=checkpoint_dir / "step_000001.pt",
        model=model,
        optimizer=optimizer,
        step=1,
        training_config=training_config,
        wandb_run_id="wandb-run-123",
        wandb_project="llm-from-scratch",
        wandb_entity="research-team",
        run_config=run_config,
    )
    train(
        model=model,
        optimizer=optimizer,
        train_batches=train_loader,
        config=training_config,
        device="cpu",
        validation_batches=validation_loader,
        start_step=1,
    )
    expected_final_state = {
        name: parameter.detach().clone() for name, parameter in model.state_dict().items()
    }

    command_arguments = [
        "--manifest",
        str(manifest_path),
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--resume",
        "latest",
        "--device",
        "cpu",
        "--precision",
        "fp32",
        "--d-model",
        "8",
        "--n-layers",
        "1",
        "--n-heads",
        "2",
        "--seq-len",
        "4",
        "--dropout",
        "0.2",
        "--batch-size",
        "2",
        "--max-steps",
        "2",
        "--grad-accum-steps",
        "1",
        "--warmup-steps",
        "0",
        "--eval-interval",
        "1",
        "--eval-batches",
        "1",
        "--sample-interval",
        "1",
        "--max-new-tokens",
        "1",
        "--checkpoint-interval",
        "1",
        "--keep-last-checkpoints",
        "1",
        "--wandb-project",
        "llm-from-scratch",
        "--wandb-entity",
        "research-team",
        "--wandb-mode",
        "disabled",
    ]
    exit_code = main(command_arguments)

    assert exit_code == 0
    final_checkpoint = torch.load(checkpoint_dir / "step_000002.pt", weights_only=True)
    assert final_checkpoint["step"] == 2
    assert final_checkpoint["wandb_run_id"] == "wandb-run-123"
    assert final_checkpoint["wandb_project"] == "llm-from-scratch"
    assert final_checkpoint["wandb_entity"] == "research-team"
    for name, expected_parameter in expected_final_state.items():
        torch.testing.assert_close(
            final_checkpoint["model_state"][name],
            expected_parameter,
            rtol=0.0,
            atol=0.0,
        )
    assert sorted(path.name for path in checkpoint_dir.glob("step_*.pt")) == ["step_000002.pt"]

    mismatched_arguments = list(command_arguments)
    project_index = mismatched_arguments.index("--wandb-project") + 1
    mismatched_arguments[project_index] = "another-project"
    with pytest.raises(ValueError, match="wandb_project does not match"):
        main(mismatched_arguments)

    mismatched_arguments = list(command_arguments)
    resume_index = mismatched_arguments.index("--resume") + 1
    mismatched_arguments[resume_index] = str(checkpoint_dir / "step_000002.pt")
    mismatched_arguments.extend(["--seed", "43"])
    with pytest.raises(ValueError, match="checkpoint run_config does not match"):
        main(mismatched_arguments)
