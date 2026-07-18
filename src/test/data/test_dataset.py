import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from data.dataset import PretrainingDataset

ShardSpec = tuple[str, str, Sequence[int]]


def _write_prepared_dataset(root: Path, shards: Sequence[ShardSpec]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, object]] = []
    split_counts: dict[str, dict[str, int]] = {}

    for relative_name, split, token_values in shards:
        shard_path = root / relative_name
        shard_path.parent.mkdir(parents=True, exist_ok=True)
        tokens = np.asarray(token_values, dtype=np.uint16)
        tokens.tofile(shard_path)
        entries.append(
            {
                "file": relative_name,
                "split": split,
                "tokens": int(tokens.size),
            }
        )
        counts = split_counts.setdefault(split, {"shards": 0, "tokens": 0})
        counts["shards"] += 1
        counts["tokens"] += int(tokens.size)

    manifest = {
        "format_version": 2,
        "storage": {"dtype": "uint16", "shard_size": 100},
        "splits": split_counts,
        "shards": entries,
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def test_dataset_returns_shifted_sequences_across_shards(tmp_path: Path) -> None:
    manifest_path = _write_prepared_dataset(
        tmp_path,
        [
            ("train/shard_0000.bin", "train", [10, 11, 12, 13, 14, 15, 16]),
            ("validation/shard_0000.bin", "validation", [100, 101, 102, 103]),
            ("train/shard_0001.bin", "train", [20, 21, 22, 23]),
        ],
    )

    dataset = PretrainingDataset(manifest_path, split="train", seq_len=3)

    assert len(dataset) == 3
    expected_sequences = [
        ([10, 11, 12], [11, 12, 13]),
        ([13, 14, 15], [14, 15, 16]),
        ([20, 21, 22], [21, 22, 23]),
    ]
    for index, (expected_inputs, expected_targets) in enumerate(expected_sequences):
        input_ids, targets = dataset[index]
        assert input_ids.dtype == torch.long
        assert targets.dtype == torch.long
        assert input_ids.tolist() == expected_inputs
        assert targets.tolist() == expected_targets


def test_dataset_batches_sequences_with_a_dataloader(tmp_path: Path) -> None:
    manifest_path = _write_prepared_dataset(
        tmp_path,
        [("shard_0000.bin", "train", list(range(13)))],
    )
    dataset = PretrainingDataset(manifest_path, split="train", seq_len=3)

    input_ids, targets = next(
        iter(DataLoader(dataset, batch_size=2, shuffle=False, drop_last=True))
    )

    assert input_ids.shape == (2, 3)
    assert targets.shape == (2, 3)
    assert input_ids.tolist() == [[0, 1, 2], [3, 4, 5]]
    assert targets.tolist() == [[1, 2, 3], [4, 5, 6]]


@pytest.mark.parametrize("seq_len", [0, -1, True])
def test_dataset_rejects_invalid_sequence_lengths(tmp_path: Path, seq_len: int) -> None:
    with pytest.raises(ValueError, match="seq_len must be a positive integer"):
        PretrainingDataset(tmp_path / "missing.json", split="train", seq_len=seq_len)


def test_dataset_rejects_a_missing_manifest(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Manifest file not found"):
        PretrainingDataset(tmp_path / "missing.json", split="train", seq_len=3)


def test_dataset_rejects_invalid_manifest_json(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("not json", encoding="utf-8")

    with pytest.raises(ValueError, match="Failed to parse manifest JSON"):
        PretrainingDataset(manifest_path, split="train", seq_len=3)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("format_version", 1, "Unsupported manifest format_version"),
        ("storage", {"dtype": "uint32"}, "Unsupported storage dtype"),
    ],
)
def test_dataset_rejects_incompatible_manifests(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    manifest_path = _write_prepared_dataset(
        tmp_path,
        [("shard_0000.bin", "train", [1, 2, 3, 4])],
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[field] = value
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        PretrainingDataset(manifest_path, split="train", seq_len=3)


def test_dataset_rejects_an_unknown_split(tmp_path: Path) -> None:
    manifest_path = _write_prepared_dataset(
        tmp_path,
        [("shard_0000.bin", "train", [1, 2, 3, 4])],
    )

    with pytest.raises(ValueError, match="Split 'validation' not found"):
        PretrainingDataset(manifest_path, split="validation", seq_len=3)


def test_dataset_rejects_a_shard_with_the_wrong_size(tmp_path: Path) -> None:
    manifest_path = _write_prepared_dataset(
        tmp_path,
        [("shard_0000.bin", "train", [1, 2, 3, 4])],
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["shards"][0]["tokens"] = 5
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="should contain 10 bytes, but contains 8 bytes"):
        PretrainingDataset(manifest_path, split="train", seq_len=3)


def test_dataset_rejects_a_shard_path_outside_the_dataset(tmp_path: Path) -> None:
    dataset_directory = tmp_path / "dataset"
    outside_shard = tmp_path / "outside.bin"
    np.asarray([1, 2, 3, 4], dtype=np.uint16).tofile(outside_shard)
    manifest_path = _write_prepared_dataset(
        dataset_directory,
        [("placeholder.bin", "train", [1, 2, 3, 4])],
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["shards"][0]["file"] = "../outside.bin"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="Shard path escapes the dataset directory"):
        PretrainingDataset(manifest_path, split="train", seq_len=3)


def test_dataset_rejects_a_split_without_complete_sequences(tmp_path: Path) -> None:
    manifest_path = _write_prepared_dataset(
        tmp_path,
        [("shard_0000.bin", "train", [1, 2, 3])],
    )

    with pytest.raises(ValueError, match="contains no complete sequences"):
        PretrainingDataset(manifest_path, split="train", seq_len=3)


@pytest.mark.parametrize("index", [-1, 2])
def test_dataset_rejects_out_of_range_indices(tmp_path: Path, index: int) -> None:
    manifest_path = _write_prepared_dataset(
        tmp_path,
        [("shard_0000.bin", "train", [1, 2, 3, 4, 5, 6, 7])],
    )
    dataset = PretrainingDataset(manifest_path, split="train", seq_len=3)

    with pytest.raises(IndexError, match="out of range"):
        dataset[index]
