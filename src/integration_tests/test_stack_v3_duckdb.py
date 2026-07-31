import hashlib
import os
from pathlib import Path

import pytest

from data.dataset import PretrainingDataset
from data.manifest import load_manifest
from data.prepare_data import load_streaming_duckdb_dataset, prepare_streaming_dataset

STACK_V3_SHARD_ENVIRONMENT_VARIABLE = "STACK_V3_SHARD_66_PATH"
STACK_V3_SHARD_FILENAME = "part-00066-50e95205-4aec-46cc-bde2-02f09aa216ac-c000.snappy.parquet"
STACK_V3_SHARD_SHA256 = "80ba520812d0f37477f726a0976e3e8063be997238c761e5b1cb873d36dddd37"


def test_duckdb_reads_the_oversized_stack_v3_shard_and_publishes_trainable_data(
    tmp_path: Path,
) -> None:
    configured_path = os.environ.get(STACK_V3_SHARD_ENVIRONMENT_VARIABLE)
    if configured_path is None:
        pytest.skip(f"set {STACK_V3_SHARD_ENVIRONMENT_VARIABLE} to the pinned Stack v3 shard")

    source_path = Path(configured_path)
    assert source_path.name == STACK_V3_SHARD_FILENAME
    with source_path.open("rb") as source_file:
        assert hashlib.file_digest(source_file, "sha256").hexdigest() == STACK_V3_SHARD_SHA256

    documents = load_streaming_duckdb_dataset(
        dataset_name=str(source_path),
        text_field="content",
        records_field="files",
        record_filters=(("language", "Python"), ("license_type", "permissive")),
        threads=2,
    )
    contents = [document["content"] for document in documents]
    assert len(contents) == 457
    assert sum(len(content) for content in contents if isinstance(content, str)) == 2_429_522

    output_dir = tmp_path / "prepared"
    manifest = prepare_streaming_dataset(
        output_dir=output_dir,
        dataset_name=str(source_path),
        source_reader="duckdb",
        records_field="files",
        record_filter="language=Python,license_type=permissive",
        text_field="content",
        num_tokens=10_000,
        shard_size=2_500,
        min_chars=64,
        workers=2,
        source_workers=2,
    )

    assert manifest["counts"]["tokens"] == 10_000
    assert manifest["dataset"]["source_reader"] == "duckdb"
    load_manifest(output_dir / "manifest.json")
    training_dataset = PretrainingDataset(
        output_dir / "manifest.json",
        split="train",
        seq_len=128,
    )
    input_ids, target_ids = training_dataset[0]
    assert input_ids[1:].tolist() == target_ids[:-1].tolist()
