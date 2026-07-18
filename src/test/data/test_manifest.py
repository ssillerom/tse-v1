import json
from pathlib import Path

import numpy as np
import pytest

from data.manifest import DataManifest, load_manifest, write_manifest


def _manifest_payload() -> dict[str, object]:
    return {
        "format_version": 2,
        "storage": {"dtype": "uint16", "shard_size": 4},
        "splits": {"train": {"tokens": 4, "shards": 1}},
        "shards": [{"file": "train.bin", "split": "train", "tokens": 4}],
    }


def test_manifest_round_trip_exposes_validated_shards(tmp_path: Path) -> None:
    np.asarray([10, 11, 12, 13], dtype=np.uint16).tofile(tmp_path / "train.bin")
    manifest_path = tmp_path / "manifest.json"

    write_manifest(manifest_path, _manifest_payload())
    manifest = load_manifest(manifest_path)

    assert isinstance(manifest, DataManifest)
    assert manifest.available_splits == frozenset({"train"})
    assert manifest.shards_for_split("train")[0].path == tmp_path / "train.bin"
    assert manifest.shards_for_split("train")[0].token_count == 4
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == _manifest_payload()


def test_writer_rejects_a_shard_from_an_undeclared_split(tmp_path: Path) -> None:
    np.asarray([10, 11, 12, 13], dtype=np.uint16).tofile(tmp_path / "train.bin")
    payload = _manifest_payload()
    payload["shards"] = [{"file": "train.bin", "split": "validation", "tokens": 4}]

    with pytest.raises(ValueError, match="undeclared split"):
        write_manifest(tmp_path / "manifest.json", payload)

    assert not (tmp_path / "manifest.json").exists()
