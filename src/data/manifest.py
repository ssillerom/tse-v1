"""Read, validate, and atomically write prepared-data manifests."""

import hashlib
import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

FORMAT_VERSION = 3
LEGACY_FORMAT_VERSION = 2
SUPPORTED_FORMAT_VERSIONS = frozenset({LEGACY_FORMAT_VERSION, FORMAT_VERSION})
STORAGE_DTYPE = "uint16"
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class ManifestShard:
    """One validated shard declared by a prepared-data manifest."""

    path: Path
    split: str
    token_count: int
    sha256: str | None


@dataclass(frozen=True)
class ManifestTokenizer:
    """Tokenizer metadata required to decode and evaluate model outputs."""

    encoding_name: str
    eot_token_id: int


@dataclass(frozen=True)
class DataManifest:
    """Validated manifest metadata needed by dataset consumers."""

    path: Path
    shards: tuple[ManifestShard, ...]
    available_splits: frozenset[str]
    tokenizer: ManifestTokenizer | None

    def shards_for_split(self, split: str) -> tuple[ManifestShard, ...]:
        """Return the shards belonging to a declared split."""
        if split not in self.available_splits:
            available = ", ".join(sorted(self.available_splits))
            raise ValueError(f"Split {split!r} not found. Available splits: {available}")

        shards = tuple(shard for shard in self.shards if shard.split == split)
        if not shards:
            raise ValueError(f"No shards found for split {split!r}")
        return shards


def load_manifest(path: str | Path) -> DataManifest:
    """Load a manifest and validate its schema, files, and checksums by default."""
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Failed to parse manifest JSON: {error} in {manifest_path}") from error

    if not isinstance(payload, dict):
        raise ValueError(f"Manifest JSON of {manifest_path} must be an object")

    return _validate_manifest(payload, manifest_path)


def write_manifest(path: str | Path, payload: Mapping[str, object]) -> None:
    """Validate and atomically write a prepared-data manifest."""
    manifest_path = Path(path)
    normalized_payload = dict(payload)
    if normalized_payload.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"Writer requires manifest format_version {FORMAT_VERSION}")
    _validate_manifest(normalized_payload, manifest_path)

    temporary_path = manifest_path.with_suffix(".json.tmp")
    try:
        temporary_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_path, manifest_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _validate_manifest(
    payload: dict[str, Any],
    manifest_path: Path,
) -> DataManifest:
    format_version = payload.get("format_version")
    if format_version not in SUPPORTED_FORMAT_VERSIONS:
        raise ValueError(f"Unsupported manifest format_version: {format_version}")

    storage = payload.get("storage")
    if not isinstance(storage, dict):
        raise ValueError("Manifest must contain a storage object")

    dtype = storage.get("dtype")
    if dtype != STORAGE_DTYPE:
        raise ValueError(f"Unsupported storage dtype: {dtype!r}")

    raw_splits = payload.get("splits")
    if not isinstance(raw_splits, dict):
        raise ValueError("Manifest must contain a splits object")
    if not all(isinstance(split, str) for split in raw_splits):
        raise ValueError("Manifest split names must be strings")
    available_splits = frozenset(raw_splits)
    tokenizer = _validate_tokenizer(payload.get("tokenizer"))

    raw_shards = payload.get("shards")
    if not isinstance(raw_shards, list):
        raise ValueError("Manifest must contain a shards list")

    shards = tuple(
        _validate_shard(
            entry=entry,
            position=position,
            manifest_directory=manifest_path.parent,
            available_splits=available_splits,
            checksum_required=format_version == FORMAT_VERSION,
        )
        for position, entry in enumerate(raw_shards)
    )
    return DataManifest(
        path=manifest_path,
        shards=shards,
        available_splits=available_splits,
        tokenizer=tokenizer,
    )


def _validate_tokenizer(raw_tokenizer: object) -> ManifestTokenizer | None:
    if raw_tokenizer is None:
        return None
    if not isinstance(raw_tokenizer, dict):
        raise ValueError("Manifest tokenizer must be an object")

    encoding_name = raw_tokenizer.get("encoding")
    if not isinstance(encoding_name, str) or not encoding_name:
        raise ValueError("Manifest tokenizer must contain a non-empty encoding")
    eot_token_id = raw_tokenizer.get("eot_token")
    if (
        not isinstance(eot_token_id, int)
        or isinstance(eot_token_id, bool)
        or not 0 <= eot_token_id <= int(np.iinfo(np.uint16).max)
    ):
        raise ValueError("Manifest tokenizer must contain a uint16-compatible eot_token")
    return ManifestTokenizer(
        encoding_name=encoding_name,
        eot_token_id=eot_token_id,
    )


def _validate_shard(
    entry: object,
    position: int,
    manifest_directory: Path,
    available_splits: frozenset[str],
    checksum_required: bool,
) -> ManifestShard:
    if not isinstance(entry, dict):
        raise ValueError(f"Shard entry at position {position} is not an object")

    file_name = entry.get("file")
    if not isinstance(file_name, str) or not file_name:
        raise ValueError("Shard entry must contain a non-empty 'file' string")

    relative_path = Path(file_name)
    if relative_path.is_absolute():
        raise ValueError(f"Shard path must be relative: {file_name!r}")

    dataset_directory = manifest_directory.resolve()
    shard_path = (dataset_directory / relative_path).resolve()
    if not shard_path.is_relative_to(dataset_directory):
        raise ValueError(f"Shard path escapes the dataset directory: {file_name!r}")
    if not shard_path.is_file():
        raise FileNotFoundError(f"Shard file not found: {shard_path}")

    split = entry.get("split")
    if not isinstance(split, str) or split not in available_splits:
        raise ValueError(f"Shard entry has an undeclared split: {split!r}")

    token_count = entry.get("tokens")
    if not isinstance(token_count, int) or isinstance(token_count, bool) or token_count <= 0:
        raise ValueError(f"Invalid token count for shard {shard_path}: {token_count!r}")

    expected_bytes = token_count * np.dtype(np.uint16).itemsize
    actual_bytes = shard_path.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"Shard {shard_path} declares {token_count} tokens "
            f"and should contain {expected_bytes} bytes, but contains {actual_bytes} bytes"
        )

    raw_sha256 = entry.get("sha256")
    if raw_sha256 is None and not checksum_required:
        sha256 = None
    elif not isinstance(raw_sha256, str) or SHA256_PATTERN.fullmatch(raw_sha256) is None:
        raise ValueError(f"Shard {shard_path} must declare a lowercase SHA-256 checksum")
    else:
        sha256 = raw_sha256
        with shard_path.open("rb") as shard_file:
            actual_sha256 = hashlib.file_digest(shard_file, "sha256").hexdigest()
        if actual_sha256 != sha256:
            raise ValueError(
                f"Shard {shard_path} SHA-256 mismatch: expected {sha256}, got {actual_sha256}"
            )

    return ManifestShard(
        path=shard_path,
        split=split,
        token_count=token_count,
        sha256=sha256,
    )
