from pathlib import Path

import numpy as np

from data.manifest import FORMAT_VERSION, STORAGE_DTYPE, write_manifest
from scripts.train_v1 import main


def _write_training_manifest(directory: Path) -> Path:
    train_tokens = np.tile(np.array([100, 101, 102, 103, 104], dtype=np.uint16), 8)
    validation_tokens = np.tile(np.array([100, 101, 102, 103, 104], dtype=np.uint16), 4)
    train_path = directory / "train.bin"
    validation_path = directory / "validation.bin"
    train_tokens.tofile(train_path)
    validation_tokens.tofile(validation_path)
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
                {"file": train_path.name, "split": "train", "tokens": 40},
                {
                    "file": validation_path.name,
                    "split": "validation",
                    "tokens": 20,
                },
            ],
        },
    )
    return manifest_path


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
            "--d-model",
            "8",
            "--n-layers",
            "1",
            "--n-heads",
            "2",
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
    assert (checkpoint_dir / "step_000001.pt").is_file()
