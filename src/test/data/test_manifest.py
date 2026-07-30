import json
from pathlib import Path

import numpy as np
import pytest

from data.manifest import DataManifest, load_manifest, write_manifest


def _manifest_payload() -> dict[str, object]:
    return {
        "format_version": 3,
        "storage": {"dtype": "uint16", "shard_size": 4},
        "splits": {"train": {"tokens": 4, "shards": 1}},
        "shards": [
            {
                "file": "train.bin",
                "split": "train",
                "tokens": 4,
                "sha256": "a58214fbfec2da6f1e9fc6a2641c8a0af73fb383860180a73d4439fe31b44189",
            }
        ],
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
    assert manifest.shards_for_split("train")[0].sha256 == (
        "a58214fbfec2da6f1e9fc6a2641c8a0af73fb383860180a73d4439fe31b44189"
    )
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == _manifest_payload()


def test_manifest_exposes_tokenizer_metadata_when_declared(tmp_path: Path) -> None:
    np.asarray([10, 11, 12, 13], dtype=np.uint16).tofile(tmp_path / "train.bin")
    payload = _manifest_payload()
    payload["tokenizer"] = {"encoding": "gpt2", "eot_token": 50_256}

    write_manifest(tmp_path / "manifest.json", payload)
    manifest = load_manifest(tmp_path / "manifest.json")

    assert manifest.tokenizer is not None
    assert manifest.tokenizer.encoding_name == "gpt2"
    assert manifest.tokenizer.eot_token_id == 50_256


def test_writer_rejects_a_shard_from_an_undeclared_split(tmp_path: Path) -> None:
    np.asarray([10, 11, 12, 13], dtype=np.uint16).tofile(tmp_path / "train.bin")
    payload = _manifest_payload()
    payload["shards"] = [{"file": "train.bin", "split": "validation", "tokens": 4}]

    with pytest.raises(ValueError, match="undeclared split"):
        write_manifest(tmp_path / "manifest.json", payload)

    assert not (tmp_path / "manifest.json").exists()


def test_manifest_rejects_same_size_shard_corruption(tmp_path: Path) -> None:
    shard_path = tmp_path / "train.bin"
    np.asarray([10, 11, 12, 13], dtype=np.uint16).tofile(shard_path)
    manifest_path = tmp_path / "manifest.json"
    write_manifest(manifest_path, _manifest_payload())
    np.asarray([10, 11, 99, 13], dtype=np.uint16).tofile(shard_path)

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_manifest(manifest_path)


def test_current_manifest_requires_a_sha256_for_every_shard(tmp_path: Path) -> None:
    np.asarray([10, 11, 12, 13], dtype=np.uint16).tofile(tmp_path / "train.bin")
    payload = _manifest_payload()
    del payload["shards"][0]["sha256"]

    with pytest.raises(ValueError, match="must declare a lowercase SHA-256"):
        write_manifest(tmp_path / "manifest.json", payload)


def test_legacy_v2_manifest_remains_readable_without_checksums(tmp_path: Path) -> None:
    np.asarray([10, 11, 12, 13], dtype=np.uint16).tofile(tmp_path / "train.bin")
    payload = _manifest_payload()
    payload["format_version"] = 2
    del payload["shards"][0]["sha256"]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    manifest = load_manifest(manifest_path)

    assert manifest.shards_for_split("train")[0].sha256 is None


def test_writer_refuses_to_publish_a_legacy_manifest_without_checksums(
    tmp_path: Path,
) -> None:
    np.asarray([10, 11, 12, 13], dtype=np.uint16).tofile(tmp_path / "train.bin")
    payload = _manifest_payload()
    payload["format_version"] = 2
    del payload["shards"][0]["sha256"]

    with pytest.raises(ValueError, match="Writer requires manifest format_version 3"):
        write_manifest(tmp_path / "manifest.json", payload)
