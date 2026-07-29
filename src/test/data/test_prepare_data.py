import json
from pathlib import Path

import numpy as np
import pytest
import tiktoken

from data.prepare_data import (
    TokenShardWriter,
    build_parser,
    inspect_dataset,
    main,
    prepare_streaming_dataset,
)


def test_writer_rejects_non_positive_shard_size(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="shard_size must be positive"):
        TokenShardWriter(tmp_path, shard_size=0)


def test_writer_refuses_to_mix_new_and_existing_shards(tmp_path: Path) -> None:
    existing_shard = tmp_path / "shard_0000.bin"
    existing_shard.write_bytes(b"old training data")

    with pytest.raises(FileExistsError, match="already contains generated artifacts"):
        TokenShardWriter(tmp_path, shard_size=4)

    assert existing_shard.read_bytes() == b"old training data"


def test_prepare_overwrite_removes_only_generated_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "shard_0000.bin").write_bytes(b"old shard")
    (tmp_path / "shard_0001.bin").write_bytes(b"stale shard")
    (tmp_path / "manifest.json").write_text("{}")
    unrelated_file = tmp_path / "notes.txt"
    unrelated_file.write_text("keep me")
    similarly_named_file = tmp_path / "shard_backup.bin"
    similarly_named_file.write_text("also keep me")

    def fake_load_dataset(**kwargs: object) -> list[dict[str, str]]:
        del kwargs
        return [{"text": "replacement"}]

    monkeypatch.setattr("data.prepare_data.load_dataset", fake_load_dataset)
    prepare_streaming_dataset(
        output_dir=tmp_path,
        dataset_name="example/dataset",
        text_field="text",
        num_tokens=10,
        shard_size=10,
        overwrite=True,
    )

    assert (tmp_path / "shard_0000.bin").exists()
    assert not (tmp_path / "shard_0001.bin").exists()
    assert unrelated_file.read_text() == "keep me"
    assert similarly_named_file.read_text() == "also keep me"


def test_writer_splits_shards_and_owns_the_total_token_limit(tmp_path: Path) -> None:
    writer = TokenShardWriter(tmp_path, shard_size=3, max_total_tokens=5)

    tokens_added = writer.add_tokens([1, 2, 3, 4, 5, 6, 7])
    writer.flush()

    assert tokens_added == 5
    assert np.fromfile(tmp_path / "shard_0000.bin", dtype=np.uint16).tolist() == [
        1,
        2,
        3,
    ]
    assert np.fromfile(tmp_path / "shard_0001.bin", dtype=np.uint16).tolist() == [4, 5]


def test_writer_leaves_no_partial_shard_when_atomic_replace_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    writer = TokenShardWriter(tmp_path, shard_size=3)
    writer.add_tokens([1, 2])

    def fail_replace(source: object, destination: object) -> None:
        raise OSError(f"cannot replace {source} with {destination}")

    monkeypatch.setattr("data.prepare_data.os.replace", fail_replace)

    with pytest.raises(OSError, match="cannot replace"):
        writer.flush()

    assert not (tmp_path / "shard_0000.bin").exists()
    assert not (tmp_path / "shard_0000.bin.tmp").exists()


@pytest.mark.parametrize("invalid_token", [-1, 65_536])
def test_writer_rejects_tokens_outside_uint16(tmp_path: Path, invalid_token: int) -> None:
    writer = TokenShardWriter(tmp_path, shard_size=3)

    with pytest.raises(ValueError, match="token IDs must fit in uint16"):
        writer.add_tokens([invalid_token])


def test_inspection_forwards_data_dir_to_hugging_face(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received_arguments: dict[str, object] = {}

    def fake_load_dataset(**kwargs: object) -> tuple[()]:
        received_arguments.update(kwargs)
        return ()

    monkeypatch.setattr("data.prepare_data.load_dataset", fake_load_dataset)

    inspect_dataset(
        dataset_name="bigcode/the-stack",
        split="train",
        data_dir="data/python",
        num_samples=0,
    )

    assert received_arguments["data_dir"] == "data/python"


@pytest.mark.parametrize(
    ("invalid_argument", "invalid_value", "message"),
    [
        ("num_tokens", 0, "num_tokens must be positive"),
        ("shard_size", 0, "shard_size must be positive"),
        ("min_chars", -1, "min_chars cannot be negative"),
        ("log_every_docs", 0, "log_every_docs must be positive"),
        ("max_docs", 0, "max_docs must be positive"),
        (
            "validation_ratio",
            -0.1,
            "validation_ratio must be greater than or equal to 0 and less than 1",
        ),
        (
            "validation_ratio",
            1.0,
            "validation_ratio must be greater than or equal to 0 and less than 1",
        ),
    ],
)
def test_prepare_rejects_invalid_limits_before_loading_the_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invalid_argument: str,
    invalid_value: int | float,
    message: str,
) -> None:
    def fail_if_called(**kwargs: object) -> tuple[()]:
        raise AssertionError(f"load_dataset called with {kwargs}")

    monkeypatch.setattr("data.prepare_data.load_dataset", fail_if_called)
    arguments: dict[str, object] = {
        "output_dir": str(tmp_path),
        "dataset_name": "example/dataset",
        "text_field": "text",
        "num_tokens": 10,
        "shard_size": 4,
        "min_chars": 0,
        "log_every_docs": 100,
    }
    arguments[invalid_argument] = invalid_value

    with pytest.raises(ValueError, match=message):
        prepare_streaming_dataset(**arguments)


def test_prepare_writes_a_reproducible_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    received_arguments: dict[str, object] = {}

    def fake_load_dataset(**kwargs: object) -> list[dict[str, object]]:
        received_arguments.update(kwargs)
        return [{"text": "hello"}, {"text": 123}, {"text": None}]

    monkeypatch.setattr("data.prepare_data.load_dataset", fake_load_dataset)

    prepare_streaming_dataset(
        output_dir=str(tmp_path),
        dataset_name="example/dataset",
        text_field="text",
        num_tokens=10,
        shard_size=10,
        revision="fixed-revision",
        encoding_name="gpt2",
    )

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest == {
        "format_version": 2,
        "dataset": {
            "path": "example/dataset",
            "name": None,
            "data_dir": None,
            "split": "train",
            "revision": "fixed-revision",
            "text_field": "text",
        },
        "limits": {"max_docs": None, "max_tokens": 10},
        "partitioning": {
            "seed": 42,
            "strategy": "none",
            "validation_ratio": 0.0,
        },
        "tokenizer": {"encoding": "gpt2", "eot_token": 50256},
        "storage": {"dtype": "uint16", "shard_size": 10},
        "counts": {
            "tokens": 2,
            "shards": 1,
            "docs_seen": 3,
            "docs_used": 1,
            "docs_skipped": 2,
            "docs_truncated": 0,
        },
        "splits": {
            "train": {
                "docs_truncated": 0,
                "docs_used": 1,
                "shards": 1,
                "tokens": 2,
            }
        },
        "shards": [{"file": "shard_0000.bin", "split": "train", "tokens": 2}],
    }
    assert received_arguments["revision"] == "fixed-revision"


def test_prepare_can_flatten_and_filter_nested_source_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_load_dataset(**kwargs: object) -> list[dict[str, object]]:
        del kwargs
        return [
            {
                "repo_path": "example/project",
                "files": [
                    {
                        "language": "Python",
                        "license_type": "permissive",
                        "content": "def add(a, b):\n    return a + b\n",
                    },
                    {
                        "language": "JavaScript",
                        "license_type": "permissive",
                        "content": "const add = (a, b) => a + b;\n",
                    },
                    {
                        "language": "Python",
                        "license_type": "permissive",
                        "content": "print(add(2, 3))\n",
                    },
                    {
                        "language": "Python",
                        "license_type": "no_license",
                        "content": "print('exclude unclear licensing')\n",
                    },
                ],
            }
        ]

    monkeypatch.setattr("data.prepare_data.load_dataset", fake_load_dataset)

    manifest = prepare_streaming_dataset(
        output_dir=tmp_path,
        dataset_name="HuggingFaceCode/stack-v3-train",
        records_field="files",
        record_filter="language=Python,license_type=permissive",
        text_field="content",
        num_tokens=1_000,
        shard_size=1_000,
    )

    assert manifest["dataset"] == {
        "path": "HuggingFaceCode/stack-v3-train",
        "name": None,
        "data_dir": None,
        "split": "train",
        "revision": None,
        "records_field": "files",
        "record_filter": "language=Python,license_type=permissive",
        "text_field": "content",
    }
    assert manifest["counts"]["docs_seen"] == 2
    assert manifest["counts"]["docs_used"] == 2

    encoding = tiktoken.get_encoding("gpt2")
    token_ids = np.fromfile(tmp_path / "shard_0000.bin", dtype=np.uint16).tolist()
    documents: list[str] = []
    document_tokens: list[int] = []
    for token_id in token_ids:
        if token_id == encoding.eot_token:
            documents.append(encoding.decode(document_tokens))
            document_tokens = []
        else:
            document_tokens.append(token_id)
    assert documents == [
        "def add(a, b):\n    return a + b\n",
        "print(add(2, 3))\n",
    ]


def test_prepare_rejects_an_unknown_nested_filter_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_load_dataset(**kwargs: object) -> list[dict[str, object]]:
        del kwargs
        return [{"files": [{"language": "Python", "content": "print('hello')"}]}]

    monkeypatch.setattr("data.prepare_data.load_dataset", fake_load_dataset)

    with pytest.raises(KeyError, match="license_type"):
        prepare_streaming_dataset(
            output_dir=tmp_path,
            dataset_name="example/nested",
            records_field="files",
            record_filter="language=Python, license_type=permissive",
            text_field="content",
            num_tokens=100,
            shard_size=100,
        )


def test_prepare_rejects_a_malformed_nested_record_filter(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="FIELD=VALUE"):
        prepare_streaming_dataset(
            output_dir=tmp_path,
            dataset_name="example/nested",
            records_field="files",
            record_filter="language=Python,",
            text_field="content",
            num_tokens=100,
            shard_size=100,
        )


def test_prepare_limits_documents_and_creates_a_deterministic_validation_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    documents = [{"text": f"document number {index}"} for index in range(100)]

    def fake_load_dataset(**kwargs: object) -> list[dict[str, str]]:
        del kwargs
        return documents

    monkeypatch.setattr("data.prepare_data.load_dataset", fake_load_dataset)
    output_dirs = [tmp_path / "first", tmp_path / "second"]

    for output_dir in output_dirs:
        prepare_streaming_dataset(
            output_dir=output_dir,
            dataset_name="example/dataset",
            text_field="text",
            num_tokens=10_000,
            shard_size=20,
            max_docs=50,
            validation_ratio=0.2,
            split_seed=7,
        )

    first_manifest = json.loads((output_dirs[0] / "manifest.json").read_text())
    second_manifest = json.loads((output_dirs[1] / "manifest.json").read_text())

    assert first_manifest == second_manifest
    assert first_manifest["counts"]["docs_seen"] == 50
    assert first_manifest["counts"]["docs_used"] == 50
    assert first_manifest["partitioning"] == {
        "seed": 7,
        "strategy": "content_hash",
        "validation_ratio": 0.2,
    }
    assert first_manifest["splits"]["train"]["docs_used"] > 0
    assert first_manifest["splits"]["validation"]["docs_used"] > 0

    for shard in first_manifest["shards"]:
        relative_path = Path(shard["file"])
        assert relative_path.parts[0] == shard["split"]
        assert (output_dirs[0] / relative_path).exists()
        assert (output_dirs[0] / relative_path).read_bytes() == (
            output_dirs[1] / relative_path
        ).read_bytes()


def test_prepare_preserves_the_source_split_without_partitioning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_load_dataset(**kwargs: object) -> list[dict[str, str]]:
        del kwargs
        return [{"text": "hello"}]

    monkeypatch.setattr("data.prepare_data.load_dataset", fake_load_dataset)

    manifest = prepare_streaming_dataset(
        output_dir=tmp_path,
        dataset_name="example/dataset",
        text_field="text",
        num_tokens=10,
        shard_size=10,
        split="validation",
    )

    assert manifest["shards"] == [{"file": "shard_0000.bin", "split": "validation", "tokens": 2}]
    assert manifest["splits"] == {
        "validation": {
            "docs_truncated": 0,
            "docs_used": 1,
            "shards": 1,
            "tokens": 2,
        }
    }


def test_prepare_preserves_previous_dataset_when_replacement_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old_shard = tmp_path / "shard_0000.bin"
    old_manifest = tmp_path / "manifest.json"
    old_shard.write_bytes(b"known-good-data")
    old_manifest.write_text('{"status": "known-good"}')

    def fake_load_dataset(**kwargs: object) -> object:
        del kwargs

        def documents() -> object:
            yield {"text": "a document"}
            raise RuntimeError("stream interrupted")

        return documents()

    monkeypatch.setattr("data.prepare_data.load_dataset", fake_load_dataset)

    with pytest.raises(RuntimeError, match="stream interrupted"):
        prepare_streaming_dataset(
            output_dir=tmp_path,
            dataset_name="example/dataset",
            text_field="text",
            num_tokens=10,
            shard_size=1,
            overwrite=True,
        )

    assert old_shard.read_bytes() == b"known-good-data"
    assert old_manifest.read_text() == '{"status": "known-good"}'


def test_prepare_restores_previous_dataset_when_installation_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old_shard = tmp_path / "shard_0000.bin"
    old_manifest = tmp_path / "manifest.json"
    old_shard.write_bytes(b"known-good-data")
    old_manifest.write_text('{"status": "known-good"}')

    def fake_load_dataset(**kwargs: object) -> list[dict[str, str]]:
        del kwargs
        return [{"text": "replacement"}]

    real_replace = __import__("os").replace

    def fail_staged_shard_install(source: object, destination: object) -> None:
        source_path = Path(source)
        destination_path = Path(destination)
        if (
            source_path.parent.name.startswith(".prepare-data-staging-")
            and destination_path.parent == tmp_path
            and destination_path.name.startswith("shard_")
        ):
            raise OSError("staged shard install failed")
        real_replace(source, destination)

    monkeypatch.setattr("data.prepare_data.load_dataset", fake_load_dataset)
    monkeypatch.setattr("data.prepare_data.os.replace", fail_staged_shard_install)

    with pytest.raises(OSError, match="staged shard install failed"):
        prepare_streaming_dataset(
            output_dir=tmp_path,
            dataset_name="example/dataset",
            text_field="text",
            num_tokens=10,
            shard_size=10,
            overwrite=True,
        )

    assert old_shard.read_bytes() == b"known-good-data"
    assert old_manifest.read_text() == '{"status": "known-good"}'
    assert not any(path.name.startswith(".prepare-data-") for path in tmp_path.iterdir())


def test_prepare_restores_previous_split_dataset_when_installation_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old_train = tmp_path / "train"
    old_validation = tmp_path / "validation"
    old_train.mkdir()
    old_validation.mkdir()
    (old_train / "shard_0000.bin").write_bytes(b"old-train")
    (old_validation / "shard_0000.bin").write_bytes(b"old-validation")
    old_manifest = tmp_path / "manifest.json"
    old_manifest.write_text('{"status": "known-good"}')

    def fake_load_dataset(**kwargs: object) -> list[dict[str, str]]:
        del kwargs
        return [{"text": f"document {index}"} for index in range(20)]

    real_replace = __import__("os").replace

    def fail_validation_directory_install(source: object, destination: object) -> None:
        source_path = Path(source)
        destination_path = Path(destination)
        if (
            source_path.parent.name.startswith(".prepare-data-staging-")
            and source_path.name == "validation"
            and destination_path == old_validation
        ):
            raise OSError("validation directory install failed")
        real_replace(source, destination)

    monkeypatch.setattr("data.prepare_data.load_dataset", fake_load_dataset)
    monkeypatch.setattr("data.prepare_data.os.replace", fail_validation_directory_install)

    with pytest.raises(OSError, match="validation directory install failed"):
        prepare_streaming_dataset(
            output_dir=tmp_path,
            dataset_name="example/dataset",
            text_field="text",
            num_tokens=1_000,
            shard_size=20,
            validation_ratio=0.5,
            overwrite=True,
        )

    assert (old_train / "shard_0000.bin").read_bytes() == b"old-train"
    assert (old_validation / "shard_0000.bin").read_bytes() == b"old-validation"
    assert old_manifest.read_text() == '{"status": "known-good"}'
    assert not any(path.name.startswith(".prepare-data-") for path in tmp_path.iterdir())


def test_prepare_cleans_up_after_atomic_manifest_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_load_dataset(**kwargs: object) -> list[dict[str, str]]:
        del kwargs
        return [{"text": "hello"}]

    real_replace = __import__("os").replace

    def fail_manifest_replace(source: object, destination: object) -> None:
        if Path(destination).name == "manifest.json":
            raise OSError("manifest replace failed")
        real_replace(source, destination)

    monkeypatch.setattr("data.prepare_data.load_dataset", fake_load_dataset)
    monkeypatch.setattr("data.manifest.os.replace", fail_manifest_replace)

    with pytest.raises(OSError, match="manifest replace failed"):
        prepare_streaming_dataset(
            output_dir=tmp_path,
            dataset_name="example/dataset",
            text_field="text",
            num_tokens=10,
            shard_size=10,
        )

    assert list(tmp_path.iterdir()) == []


def test_cli_exposes_prepare_configuration() -> None:
    arguments = build_parser().parse_args(
        [
            "prepare",
            "--dataset-name",
            "example/dataset",
            "--output-dir",
            "data/example",
            "--text-field",
            "text",
            "--records-field",
            "files",
            "--record-filter",
            "language=Python,license_type=permissive",
            "--num-tokens",
            "100",
            "--revision",
            "fixed",
            "--encoding",
            "gpt2",
            "--max-docs",
            "50",
            "--validation-ratio",
            "0.1",
            "--split-seed",
            "7",
            "--overwrite",
        ]
    )

    assert (
        arguments.command,
        arguments.num_tokens,
        arguments.revision,
        arguments.encoding,
        arguments.max_docs,
        arguments.validation_ratio,
        arguments.split_seed,
        arguments.overwrite,
        arguments.records_field,
        arguments.record_filter,
    ) == (
        "prepare",
        100,
        "fixed",
        "gpt2",
        50,
        0.1,
        7,
        True,
        "files",
        "language=Python,license_type=permissive",
    )


def test_cli_prepare_executes_with_parsed_arguments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_load_dataset(**kwargs: object) -> list[dict[str, str]]:
        del kwargs
        return [{"body": "hello"}]

    monkeypatch.setattr("data.prepare_data.load_dataset", fake_load_dataset)

    exit_code = main(
        [
            "prepare",
            "--dataset-name",
            "example/dataset",
            "--output-dir",
            str(tmp_path),
            "--text-field",
            "body",
            "--num-tokens",
            "10",
            "--shard-size",
            "10",
            "--revision",
            "fixed",
        ]
    )

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert exit_code == 0
    assert manifest["dataset"]["text_field"] == "body"
    assert manifest["dataset"]["revision"] == "fixed"
