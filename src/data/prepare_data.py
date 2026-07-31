"""Pre-tokenize streaming Hugging Face datasets into uint16 binary shards.

Documents are tokenized with a configurable tiktoken encoding and separated by
that encoding's end-of-text token. Each run also writes a manifest describing
the source, tokenizer, storage format, counters, and generated shards.
"""

import argparse
import hashlib
import logging
import os
import re
import shutil
import tempfile
import time
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import cast

import datasets as hf_datasets  # type: ignore[import-untyped]
import duckdb
import numpy as np
import tiktoken
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils.tqdm import disable_progress_bars as disable_hf_progress_bars
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset

from src.data.manifest import FORMAT_VERSION, STORAGE_DTYPE, write_manifest

LOGGER = logging.getLogger(__name__)
disable_datasets_progress_bars = hf_datasets.disable_progress_bars
load_dataset = hf_datasets.load_dataset
UINT16_MAX = int(np.iinfo(np.uint16).max)
SHARD_FILENAME_PATTERN = re.compile(r"shard_\d{4,}\.bin(?:\.tmp)?\Z")
GENERATED_DIRECTORY_NAMES = frozenset({"train", "validation"})
TOKENIZATION_BATCH_MAX_DOCS = 256
TOKENIZATION_BATCH_MAX_CHARS = 4_000_000
DUCKDB_BATCH_SIZE = 256
SOURCE_READERS = frozenset({"hugging_face", "duckdb"})


def _find_generated_artifacts(output_dir: Path) -> list[Path]:
    if not output_dir.exists():
        return []
    return sorted(
        (
            path
            for path in output_dir.iterdir()
            if (
                path.is_dir()
                and path.name in GENERATED_DIRECTORY_NAMES
                or path.is_file()
                and (
                    SHARD_FILENAME_PATTERN.fullmatch(path.name)
                    or path.name in {"manifest.json", "manifest.json.tmp"}
                )
            )
        ),
        key=lambda path: path.name,
    )


@dataclass(frozen=True)
class ShardMetadata:
    """Metadata recorded for one generated shard."""

    file: str
    tokens: int
    sha256: str


@dataclass(frozen=True)
class PreparationStats:
    """Counters reported only after a prepared dataset is published."""

    tokens: int
    shards: int
    docs_seen: int
    docs_used: int
    docs_skipped: int
    docs_truncated: int
    split_tokens: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class DuckDBParquetSource:
    """Ordered local files or files pinned to one Hugging Face dataset revision."""

    files: tuple[str, ...]
    repo_id: str | None = None
    revision: str | None = None
    hf_token: bool = False


class TokenShardWriter:
    """Write bounded token sequences to atomic uint16 shard files."""

    def __init__(
        self,
        output_dir: str | os.PathLike[str],
        shard_size: int,
        max_total_tokens: int | None = None,
    ) -> None:
        if shard_size <= 0:
            raise ValueError("shard_size must be positive")
        if max_total_tokens is not None and max_total_tokens <= 0:
            raise ValueError("max_total_tokens must be positive")

        self.output_dir = Path(output_dir)
        self.shard_size = shard_size
        self.max_total_tokens = max_total_tokens
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_output_directory_is_empty()

        self.buffer = np.empty(self.shard_size, dtype=np.uint16)
        self.pos = 0
        self.shard_idx = 0
        self.total_tokens = 0
        self.shards: list[ShardMetadata] = []

    def _ensure_output_directory_is_empty(self) -> None:
        existing_artifacts = _find_generated_artifacts(self.output_dir)
        if existing_artifacts:
            names = ", ".join(path.name for path in existing_artifacts)
            raise FileExistsError(
                f"{self.output_dir} already contains generated artifacts: {names}. "
                "Use prepare_streaming_dataset(..., overwrite=True) for safe replacement."
            )

    def add_tokens(self, tokens: Sequence[int]) -> int:
        """Add tokens and return how many were accepted before the total limit."""
        token_count = len(tokens)
        if self.max_total_tokens is not None:
            remaining_total = self.max_total_tokens - self.total_tokens
            if remaining_total <= 0:
                return 0
            token_count = min(token_count, remaining_total)

        offset = 0
        while offset < token_count:
            remaining_shard = self.shard_size - self.pos
            take = min(remaining_shard, token_count - offset)
            chunk = np.asarray(tokens[offset : offset + take], dtype=np.int64)
            if np.any(chunk < 0) or np.any(chunk > UINT16_MAX):
                raise ValueError("token IDs must fit in uint16")

            self.buffer[self.pos : self.pos + take] = chunk.astype(np.uint16)
            self.pos += take
            self.total_tokens += take
            offset += take

            if self.pos == self.shard_size:
                self.flush()

        return token_count

    def flush(self) -> None:
        """Atomically write the buffered tokens, if any, to the next shard."""
        if self.pos == 0:
            return

        shard_path = self.output_dir / f"shard_{self.shard_idx:04d}.bin"
        temporary_path = shard_path.with_suffix(".bin.tmp")
        token_count = self.pos

        try:
            self.buffer[:token_count].tofile(temporary_path)
            os.replace(temporary_path, shard_path)
        finally:
            temporary_path.unlink(missing_ok=True)

        with shard_path.open("rb") as shard_file:
            sha256 = hashlib.file_digest(shard_file, "sha256").hexdigest()
        self.shards.append(
            ShardMetadata(
                file=shard_path.name,
                tokens=token_count,
                sha256=sha256,
            )
        )
        self.shard_idx += 1
        self.pos = 0


def load_streaming_hf_dataset(
    dataset_name: str,
    split: str = "train",
    name: str | None = None,
    data_dir: str | None = None,
    revision: str | None = None,
    hf_token: bool = False,
) -> Iterable[Mapping[str, object]]:
    """Load a Hugging Face dataset in streaming mode."""
    kwargs: dict[str, object] = {
        "path": dataset_name,
        "split": split,
        "streaming": True,
    }
    if name is not None:
        kwargs["name"] = name
    if data_dir is not None:
        kwargs["data_dir"] = data_dir
    if revision is not None:
        kwargs["revision"] = revision
    if hf_token:
        kwargs["token"] = True

    return cast(Iterable[Mapping[str, object]], load_dataset(**kwargs))


def _duckdb_parquet_files(
    dataset_name: str,
    data_dir: str | None,
    revision: str | None,
    hf_token: bool,
) -> DuckDBParquetSource:
    local_path = Path(dataset_name)
    if local_path.is_file():
        return DuckDBParquetSource(files=(str(local_path),))
    if local_path.is_dir():
        local_parquet_directory = local_path / data_dir if data_dir is not None else local_path
        parquet_paths = tuple(
            str(path) for path in sorted(local_parquet_directory.glob("*.parquet"))
        )
        if not parquet_paths:
            raise FileNotFoundError(f"no Parquet files found in {local_parquet_directory}")
        return DuckDBParquetSource(files=parquet_paths)

    remote_parquet_directory = data_dir or "data"
    prefix = f"{remote_parquet_directory.rstrip('/')}/"
    api = HfApi(token=True if hf_token else None)
    parquet_paths = tuple(
        sorted(
            path
            for path in api.list_repo_files(
                repo_id=dataset_name,
                repo_type="dataset",
                revision=revision,
            )
            if path.startswith(prefix) and path.endswith(".parquet")
        )
    )
    if not parquet_paths:
        raise FileNotFoundError(
            f"no Parquet files found for {dataset_name!r} in {remote_parquet_directory!r}"
        )
    return DuckDBParquetSource(
        files=parquet_paths,
        repo_id=dataset_name,
        revision=revision,
        hf_token=hf_token,
    )


def _download_duckdb_parquet_file(
    source: DuckDBParquetSource,
    filename: str,
    local_dir: Path,
) -> str:
    if source.repo_id is None:
        return filename
    return hf_hub_download(
        repo_id=source.repo_id,
        filename=filename,
        repo_type="dataset",
        revision=source.revision,
        token=True if source.hf_token else None,
        local_dir=local_dir,
    )


def _materialize_duckdb_parquet_files(
    source: DuckDBParquetSource,
    workers: int,
) -> Iterable[str]:
    if source.repo_id is None:
        yield from source.files
        return

    with tempfile.TemporaryDirectory(prefix="prepare-data-duckdb-") as temporary_name:
        temporary_root = Path(temporary_name)
        next_index = 0
        pending: list[tuple[int, Future[str]]] = []
        executor = ThreadPoolExecutor(max_workers=workers)

        def submit(index: int) -> tuple[int, Future[str]]:
            local_dir = temporary_root / f"{index:08d}"
            return (
                index,
                executor.submit(
                    _download_duckdb_parquet_file,
                    source,
                    source.files[index],
                    local_dir,
                ),
            )

        try:
            while next_index < min(workers, len(source.files)):
                pending.append(submit(next_index))
                next_index += 1

            while pending:
                index, future = pending.pop(0)
                try:
                    yield future.result()
                finally:
                    shutil.rmtree(temporary_root / f"{index:08d}", ignore_errors=True)

                if next_index < len(source.files):
                    pending.append(submit(next_index))
                    next_index += 1
        finally:
            executor.shutdown(wait=True, cancel_futures=True)


def _quote_duckdb_identifier(identifier: str) -> str:
    return f'"{identifier.replace('"', '""')}"'


def load_streaming_duckdb_dataset(
    dataset_name: str,
    text_field: str,
    records_field: str | None,
    record_filters: Sequence[tuple[str, str]],
    data_dir: str | None = None,
    revision: str | None = None,
    hf_token: bool = False,
    threads: int = 1,
) -> Iterable[Mapping[str, object]]:
    """Read Parquet records through DuckDB without materializing nested Arrow arrays."""
    source = _duckdb_parquet_files(
        dataset_name=dataset_name,
        data_dir=data_dir,
        revision=revision,
        hf_token=hf_token,
    )

    def documents() -> Iterable[Mapping[str, object]]:
        connection = duckdb.connect()
        try:
            connection.execute(f"SET threads = {threads}")
            connection.execute("SET preserve_insertion_order = true")
            connection.execute("SET enable_progress_bar = false")

            quoted_text_field = _quote_duckdb_identifier(text_field)
            parameters = [value for _, value in record_filters]
            for parquet_file in _materialize_duckdb_parquet_files(source, workers=threads):
                connection.from_parquet(parquet_file).create_view(
                    "__prepare_data_source",
                    replace=True,
                )
                if records_field is None:
                    query = f'SELECT {quoted_text_field} FROM "__prepare_data_source"'
                else:
                    quoted_records_field = _quote_duckdb_identifier(records_field)
                    record_column = _quote_duckdb_identifier("__record")
                    filters = " AND ".join(
                        f"{record_column}.{_quote_duckdb_identifier(field)} = ?"
                        for field, _ in record_filters
                    )
                    where_clause = f" WHERE {filters}" if filters else ""
                    query = (
                        f"SELECT {record_column}.{quoted_text_field} AS {quoted_text_field} "
                        f"FROM (SELECT unnest({quoted_records_field}) AS {record_column} "
                        'FROM "__prepare_data_source") '
                        f"{where_clause}"
                    )

                reader = connection.sql(query, params=parameters).to_arrow_reader(
                    batch_size=DUCKDB_BATCH_SIZE
                )
                for batch in reader:
                    for value in batch.column(0).to_pylist():
                        yield {text_field: value}
        finally:
            connection.close()

    return documents()


def _parse_record_filter(
    record_filter: str | None,
) -> tuple[tuple[str, str], ...]:
    if record_filter is None:
        return ()

    filters: list[tuple[str, str]] = []
    for clause in record_filter.split(","):
        field, separator, value = clause.partition("=")
        field = field.strip()
        value = value.strip()
        if not separator or not field or not value:
            raise ValueError("record_filter must use FIELD=VALUE clauses separated by commas")
        filters.append((field, value))
    return tuple(filters)


def _iter_source_documents(
    dataset: Iterable[Mapping[str, object]],
    records_field: str | None,
    record_filters: Sequence[tuple[str, str]],
) -> Iterable[Mapping[str, object]]:
    """Yield top-level documents or selected records from one nested list field."""
    if records_field is None:
        yield from dataset
        return

    for source_record in dataset:
        if records_field not in source_record:
            raise KeyError(
                f"records_field={records_field!r} not found. "
                f"Available keys: {list(source_record.keys())}"
            )
        nested_records = source_record[records_field]
        if not isinstance(nested_records, Sequence) or isinstance(nested_records, (str, bytes)):
            raise TypeError(f"records_field={records_field!r} must contain a sequence")

        for position, nested_record in enumerate(nested_records):
            if not isinstance(nested_record, Mapping):
                raise TypeError(
                    f"record {position} in records_field={records_field!r} must be a mapping"
                )
            missing_filter_fields = [
                filter_field
                for filter_field, _ in record_filters
                if filter_field not in nested_record
            ]
            if missing_filter_fields:
                raise KeyError(
                    f"filter fields {missing_filter_fields!r} not found in record {position} "
                    f"of records_field={records_field!r}. "
                    f"Available keys: {list(nested_record.keys())}"
                )
            if any(
                nested_record[filter_field] != filter_value
                for filter_field, filter_value in record_filters
            ):
                continue
            yield nested_record


def _identity_collate(document: object) -> object:
    return document


def _parallel_source_documents(
    dataset: Iterable[Mapping[str, object]],
    source_workers: int,
) -> Iterable[Mapping[str, object]]:
    if source_workers == 0:
        return dataset
    if not isinstance(dataset, TorchDataset):
        raise TypeError(
            "source_workers requires a Hugging Face or PyTorch Dataset that supports "
            "multi-process loading"
        )
    return cast(
        Iterable[Mapping[str, object]],
        DataLoader(
            dataset,
            batch_size=None,
            num_workers=source_workers,
            collate_fn=_identity_collate,
            prefetch_factor=1,
        ),
    )


def inspect_dataset(
    dataset_name: str,
    split: str = "train",
    name: str | None = None,
    data_dir: str | None = None,
    revision: str | None = None,
    hf_token: bool = False,
    num_samples: int = 5,
) -> None:
    """Print a small number of samples from a streaming dataset."""
    if num_samples < 0:
        raise ValueError("num_samples cannot be negative")

    print("=" * 80)
    print("DATASET INSPECTION")
    print("=" * 80)
    print(f"dataset_name: {dataset_name}")
    print(f"name:         {name}")
    print(f"data_dir:     {data_dir}")
    print(f"revision:     {revision}")
    print(f"split:        {split}")
    print("=" * 80)

    dataset = load_streaming_hf_dataset(
        dataset_name=dataset_name,
        split=split,
        name=name,
        data_dir=data_dir,
        revision=revision,
        hf_token=hf_token,
    )
    for idx, document in enumerate(islice(dataset, num_samples)):
        print(f"\n--- Example {idx} ---")
        print("keys:", list(document.keys()))
        for key, value in document.items():
            if isinstance(value, str):
                preview = value[:500].replace("\n", "\\n")
                print(f"{key}: {preview!r}")
            else:
                print(f"{key}: {type(value).__name__} = {value}")


def _validate_preparation_limits(
    num_tokens: int,
    shard_size: int,
    min_chars: int,
    log_every_docs: int,
    workers: int,
    source_workers: int,
    source_reader: str,
    max_docs: int | None,
    validation_ratio: float,
) -> None:
    if num_tokens <= 0:
        raise ValueError("num_tokens must be positive")
    if shard_size <= 0:
        raise ValueError("shard_size must be positive")
    if min_chars < 0:
        raise ValueError("min_chars cannot be negative")
    if log_every_docs <= 0:
        raise ValueError("log_every_docs must be positive")
    if workers <= 0:
        raise ValueError("workers must be positive")
    if source_workers < 0:
        raise ValueError("source_workers cannot be negative")
    if source_reader not in SOURCE_READERS:
        choices = ", ".join(sorted(SOURCE_READERS))
        raise ValueError(f"source_reader must be one of: {choices}")
    if max_docs is not None and max_docs <= 0:
        raise ValueError("max_docs must be positive")
    if not 0.0 <= validation_ratio < 1.0:
        raise ValueError("validation_ratio must be greater than or equal to 0 and less than 1")


def _document_output_split(
    text: str,
    source_split: str,
    validation_ratio: float,
    split_seed: int,
) -> str:
    if validation_ratio == 0.0:
        return source_split

    hasher = hashlib.blake2b(digest_size=8, person=b"llmfsplit")
    hasher.update(str(split_seed).encode("ascii"))
    hasher.update(b"\0")
    hasher.update(text.encode("utf-8"))
    digest = hasher.digest()
    sample = int.from_bytes(digest, byteorder="big") / 2**64
    return "validation" if sample < validation_ratio else "train"


def _remove_artifact(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def _install_staged_artifacts(staging_dir: Path, output_dir: Path) -> None:
    staged_artifacts = [
        path for path in _find_generated_artifacts(staging_dir) if not path.name.endswith(".tmp")
    ]
    if not any(path.name == "manifest.json" for path in staged_artifacts):
        raise RuntimeError("staged dataset has no manifest")

    # Install shards first and the manifest last, so the manifest never advertises
    # shards that have not yet reached the destination.
    staged_artifacts.sort(key=lambda path: (path.name == "manifest.json", path.name))
    previous_artifacts = _find_generated_artifacts(output_dir)
    backup_dir = Path(tempfile.mkdtemp(prefix=".prepare-data-backup-", dir=output_dir))
    backed_up_names: list[str] = []
    installed_paths: list[Path] = []

    try:
        for previous_path in previous_artifacts:
            os.replace(previous_path, backup_dir / previous_path.name)
            backed_up_names.append(previous_path.name)
        for staged_path in staged_artifacts:
            destination = output_dir / staged_path.name
            os.replace(staged_path, destination)
            installed_paths.append(destination)
    except Exception as installation_error:
        for installed_path in installed_paths:
            _remove_artifact(installed_path)

        restore_failures: list[str] = []
        for name in backed_up_names:
            backup_path = backup_dir / name
            if not backup_path.exists():
                continue
            try:
                os.replace(backup_path, output_dir / name)
            except OSError:
                restore_failures.append(name)

        if restore_failures:
            failed_names = ", ".join(restore_failures)
            raise RuntimeError(
                f"failed to restore {failed_names}; recovery files remain in {backup_dir}"
            ) from installation_error

        shutil.rmtree(backup_dir)
        raise

    shutil.rmtree(backup_dir)


def _prepare_into_staging_directory(
    staging_dir: Path,
    dataset: Iterable[Mapping[str, object]],
    dataset_manifest: Mapping[str, object],
    source_split: str,
    text_field: str,
    num_tokens: int,
    shard_size: int,
    min_chars: int,
    log_every_docs: int,
    workers: int,
    max_docs: int | None,
    validation_ratio: float,
    split_seed: int,
    encoding: tiktoken.Encoding,
    encoding_name: str,
) -> tuple[dict[str, object], PreparationStats]:
    output_splits = (source_split,) if validation_ratio == 0.0 else ("train", "validation")
    split_output_dirs = (
        {source_split: staging_dir}
        if validation_ratio == 0.0
        else {
            "train": staging_dir / "train",
            "validation": staging_dir / "validation",
        }
    )
    writers = {
        split_name: TokenShardWriter(
            output_dir=split_output_dirs[split_name],
            shard_size=shard_size,
        )
        for split_name in output_splits
    }
    split_counts = {split_name: {"docs_used": 0, "docs_truncated": 0} for split_name in writers}
    docs_seen = 0
    docs_used = 0
    docs_skipped = 0
    docs_truncated = 0
    total_tokens = 0

    document_iterator = iter(dataset)
    source_exhausted = False
    source_error: Exception | None = None
    next_log_at = log_every_docs
    while total_tokens < num_tokens and not source_exhausted:
        batch_documents: list[str | None] = []
        texts: list[str] = []
        batch_chars = 0
        while (
            len(batch_documents) < TOKENIZATION_BATCH_MAX_DOCS
            and batch_chars < TOKENIZATION_BATCH_MAX_CHARS
        ):
            if max_docs is not None and docs_seen + len(batch_documents) >= max_docs:
                source_exhausted = True
                break
            try:
                document = next(document_iterator)
            except StopIteration:
                source_exhausted = True
                break
            except Exception as error:
                source_error = error
                source_exhausted = True
                break

            if text_field not in document:
                source_error = KeyError(
                    f"text_field={text_field!r} not found. Available keys: {list(document.keys())}"
                )
                source_exhausted = True
                break

            text = document[text_field]
            if not isinstance(text, str) or len(text) < min_chars:
                batch_documents.append(None)
                continue
            batch_documents.append(text)
            texts.append(text)
            batch_chars += len(text)

        encoded_documents: list[list[int]] = []
        if texts:
            if workers == 1:
                encoded_documents = [encoding.encode(text, disallowed_special=()) for text in texts]
            else:
                encoded_documents = encoding.encode_batch(
                    texts,
                    num_threads=workers,
                    disallowed_special=(),
                )

        encoded_index = 0
        for text in batch_documents:
            docs_seen += 1
            if text is None:
                docs_skipped += 1
            else:
                tokens = encoded_documents[encoded_index]
                encoded_index += 1
                tokens.append(encoding.eot_token)
                output_split = _document_output_split(
                    text,
                    source_split,
                    validation_ratio,
                    split_seed,
                )
                writer = writers[output_split]
                remaining_tokens = num_tokens - total_tokens
                document_truncated = len(tokens) > remaining_tokens
                tokens_to_write = tokens if not document_truncated else tokens[:remaining_tokens]
                if document_truncated:
                    tokens_to_write[-1] = encoding.eot_token
                tokens_added = writer.add_tokens(tokens_to_write)
                total_tokens += tokens_added
                if tokens_added > 0:
                    docs_used += 1
                    split_counts[output_split]["docs_used"] += 1
                if document_truncated:
                    docs_truncated += 1
                    split_counts[output_split]["docs_truncated"] += 1

            if docs_seen >= next_log_at:
                saved_tokens = sum(
                    shard.tokens for writer in writers.values() for shard in writer.shards
                )
                saved_shards = sum(len(writer.shards) for writer in writers.values())
                percentage = saved_tokens / num_tokens * 100
                LOGGER.info(
                    "Progress | tokens_saved=%s/%s (%.1f%%) | shards=%s",
                    f"{saved_tokens:,}",
                    f"{num_tokens:,}",
                    percentage,
                    f"{saved_shards:,}",
                )
                next_log_at += log_every_docs

            if total_tokens >= num_tokens:
                break

        if source_error is not None and total_tokens < num_tokens:
            raise source_error

    for writer in writers.values():
        writer.flush()
    total_tokens = sum(writer.total_tokens for writer in writers.values())
    total_shards = sum(writer.shard_idx for writer in writers.values())
    shards: list[dict[str, object]] = []
    splits: dict[str, object] = {}
    for split_name, writer in writers.items():
        relative_directory = Path(".") if validation_ratio == 0.0 else Path(split_name)
        shards.extend(
            {
                "file": str(relative_directory / shard.file),
                "split": split_name,
                "tokens": shard.tokens,
                "sha256": shard.sha256,
            }
            for shard in writer.shards
        )
        splits[split_name] = {
            "tokens": writer.total_tokens,
            "shards": writer.shard_idx,
            **split_counts[split_name],
        }

    manifest: dict[str, object] = {
        "format_version": FORMAT_VERSION,
        "dataset": dict(dataset_manifest),
        "limits": {"max_tokens": num_tokens, "max_docs": max_docs},
        "partitioning": {
            "strategy": "none" if validation_ratio == 0.0 else "content_hash",
            "validation_ratio": validation_ratio,
            "seed": split_seed,
        },
        "tokenizer": {
            "encoding": encoding_name,
            "eot_token": encoding.eot_token,
        },
        "storage": {"dtype": STORAGE_DTYPE, "shard_size": shard_size},
        "counts": {
            "tokens": total_tokens,
            "shards": total_shards,
            "docs_seen": docs_seen,
            "docs_used": docs_used,
            "docs_skipped": docs_skipped,
            "docs_truncated": docs_truncated,
        },
        "splits": splits,
        "shards": shards,
    }
    write_manifest(staging_dir / "manifest.json", manifest)
    stats = PreparationStats(
        tokens=total_tokens,
        shards=total_shards,
        docs_seen=docs_seen,
        docs_used=docs_used,
        docs_skipped=docs_skipped,
        docs_truncated=docs_truncated,
        split_tokens=tuple(
            (split_name, writer.total_tokens) for split_name, writer in writers.items()
        ),
    )
    return manifest, stats


def _log_preparation_summary(stats: PreparationStats, elapsed: float) -> None:
    tokens_per_second = stats.tokens / max(elapsed, 1e-9)
    split_token_stats = " | ".join(
        f"{split_name}={tokens:,}" for split_name, tokens in stats.split_tokens
    )
    elapsed_seconds = max(0, round(elapsed))
    elapsed_hours, remaining_seconds = divmod(elapsed_seconds, 3600)
    elapsed_minutes, elapsed_seconds = divmod(remaining_seconds, 60)
    elapsed_text = f"{elapsed_hours}h {elapsed_minutes:02d}m {elapsed_seconds:02d}s"
    LOGGER.info("Completed")
    LOGGER.info("Tokens | total=%s | %s", f"{stats.tokens:,}", split_token_stats)
    LOGGER.info("Shards | total=%s", f"{stats.shards:,}")
    LOGGER.info(
        "Documents | seen=%s | used=%s | skipped=%s | truncated=%s",
        f"{stats.docs_seen:,}",
        f"{stats.docs_used:,}",
        f"{stats.docs_skipped:,}",
        f"{stats.docs_truncated:,}",
    )
    LOGGER.info(
        "Performance | elapsed=%s | throughput=%s tok/s",
        elapsed_text,
        f"{tokens_per_second:,.0f}",
    )


def prepare_streaming_dataset(
    output_dir: str | os.PathLike[str],
    dataset_name: str,
    text_field: str,
    num_tokens: int,
    shard_size: int = 100_000_000,
    split: str = "train",
    name: str | None = None,
    data_dir: str | None = None,
    revision: str | None = None,
    hf_token: bool = False,
    min_chars: int = 0,
    log_every_docs: int = 10_000,
    workers: int = 1,
    source_workers: int = 0,
    source_reader: str = "hugging_face",
    max_docs: int | None = None,
    validation_ratio: float = 0.0,
    split_seed: int = 42,
    encoding_name: str = "gpt2",
    overwrite: bool = False,
    records_field: str | None = None,
    record_filter: str | None = None,
) -> dict[str, object]:
    """Stream, tokenize, and transactionally replace a dataset and its manifest."""
    _validate_preparation_limits(
        num_tokens,
        shard_size,
        min_chars,
        log_every_docs,
        workers,
        source_workers,
        source_reader,
        max_docs,
        validation_ratio,
    )
    if records_field is not None and not records_field:
        raise ValueError("records_field must be non-empty when provided")
    if records_field is None and record_filter is not None:
        raise ValueError("record_filter requires records_field")
    parsed_record_filter = _parse_record_filter(record_filter)
    if source_reader == "duckdb":
        if split != "train":
            raise ValueError("source_reader='duckdb' only supports split='train'")
        if name is not None:
            raise ValueError("source_reader='duckdb' does not support dataset configs")

    encoding = tiktoken.get_encoding(encoding_name)
    if encoding.max_token_value > UINT16_MAX:
        raise ValueError(
            f"encoding {encoding_name!r} has token IDs above the uint16 limit ({UINT16_MAX})"
        )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    existing_artifacts = _find_generated_artifacts(output_path)
    if existing_artifacts and not overwrite:
        names = ", ".join(path.name for path in existing_artifacts)
        raise FileExistsError(
            f"{output_path} already contains generated artifacts: {names}. "
            "Pass overwrite=True only if replacing them is intentional."
        )

    started_at = time.perf_counter()
    if source_reader == "duckdb":
        dataset = load_streaming_duckdb_dataset(
            dataset_name=dataset_name,
            text_field=text_field,
            records_field=records_field,
            record_filters=parsed_record_filter,
            data_dir=data_dir,
            revision=revision,
            hf_token=hf_token,
            threads=max(1, source_workers),
        )
    else:
        source_dataset = load_streaming_hf_dataset(
            dataset_name=dataset_name,
            split=split,
            name=name,
            data_dir=data_dir,
            revision=revision,
            hf_token=hf_token,
        )
        dataset = _iter_source_documents(
            dataset=_parallel_source_documents(
                dataset=source_dataset,
                source_workers=source_workers,
            ),
            records_field=records_field,
            record_filters=parsed_record_filter,
        )
    dataset_manifest: dict[str, object] = {
        "path": dataset_name,
        "name": name,
        "data_dir": data_dir,
        "split": split,
        "revision": revision,
        "text_field": text_field,
    }
    if records_field is not None:
        dataset_manifest["records_field"] = records_field
        dataset_manifest["record_filter"] = record_filter
    if source_reader != "hugging_face":
        dataset_manifest["source_reader"] = source_reader
    if source_workers > 0:
        dataset_manifest["source_workers"] = source_workers

    with tempfile.TemporaryDirectory(
        prefix=".prepare-data-staging-",
        dir=output_path,
    ) as staging_name:
        staging_path = Path(staging_name)
        try:
            manifest, stats = _prepare_into_staging_directory(
                staging_dir=staging_path,
                dataset=dataset,
                dataset_manifest=dataset_manifest,
                source_split=split,
                text_field=text_field,
                num_tokens=num_tokens,
                shard_size=shard_size,
                min_chars=min_chars,
                log_every_docs=log_every_docs,
                workers=workers,
                max_docs=max_docs,
                validation_ratio=validation_ratio,
                split_seed=split_seed,
                encoding=encoding,
                encoding_name=encoding_name,
            )
        finally:
            close_dataset = getattr(dataset, "close", None)
            if callable(close_dataset):
                close_dataset()
        _install_staged_artifacts(staging_path, output_path)

    _log_preparation_summary(stats, time.perf_counter() - started_at)
    return manifest


def _add_dataset_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-name", required=True, help="Hugging Face dataset path")
    parser.add_argument("--split", default="train")
    parser.add_argument("--name", help="Dataset config/name")
    parser.add_argument("--data-dir", help="Dataset data_dir, for example data/python")
    parser.add_argument("--revision", help="Pinned dataset revision for reproducibility")
    parser.add_argument(
        "--hf-token",
        action="store_true",
        help="Use the locally configured Hugging Face token",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="Generate binary shards")
    _add_dataset_arguments(prepare_parser)
    prepare_parser.add_argument("--output-dir", required=True)
    prepare_parser.add_argument("--text-field", required=True)
    prepare_parser.add_argument(
        "--records-field",
        help="Flatten records from this nested sequence field before reading text",
    )
    prepare_parser.add_argument(
        "--record-filter",
        help=(
            "Keep nested records matching every FIELD=VALUE clause, separated by commas; "
            "requires --records-field"
        ),
    )
    prepare_parser.add_argument("--num-tokens", required=True, type=int)
    prepare_parser.add_argument("--shard-size", type=int, default=100_000_000)
    prepare_parser.add_argument("--min-chars", type=int, default=0)
    prepare_parser.add_argument(
        "--log-every-docs",
        type=int,
        default=10_000,
        help="Emit compact token and shard progress after this many source documents",
    )
    prepare_parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of tokenizer threads used for deterministic batch encoding",
    )
    prepare_parser.add_argument(
        "--source-workers",
        type=int,
        default=0,
        help="Number of processes used to download and decode source shards",
    )
    prepare_parser.add_argument(
        "--source-reader",
        choices=sorted(SOURCE_READERS),
        default="hugging_face",
        help="Reader used for source data; duckdb avoids materializing nested Arrow arrays",
    )
    prepare_parser.add_argument(
        "--max-docs",
        type=int,
        help="Stop after reading this many source documents",
    )
    prepare_parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.0,
        help="Deterministic fraction of valid documents reserved for validation",
    )
    prepare_parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Seed used by deterministic content-hash partitioning",
    )
    prepare_parser.add_argument("--encoding", default="gpt2")
    prepare_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace generated shards and manifest in the output directory",
    )

    inspect_parser = subparsers.add_parser("inspect", help="Preview dataset records")
    _add_dataset_arguments(inspect_parser)
    inspect_parser.add_argument("--num-samples", type=int, default=5)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface."""
    args = build_parser().parse_args(argv)
    disable_datasets_progress_bars()
    disable_hf_progress_bars()
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    LOGGER.setLevel(logging.INFO)

    if args.command == "inspect":
        inspect_dataset(
            dataset_name=args.dataset_name,
            split=args.split,
            name=args.name,
            data_dir=args.data_dir,
            revision=args.revision,
            hf_token=args.hf_token,
            num_samples=args.num_samples,
        )
        return 0

    prepare_streaming_dataset(
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        text_field=args.text_field,
        num_tokens=args.num_tokens,
        records_field=args.records_field,
        record_filter=args.record_filter,
        shard_size=args.shard_size,
        split=args.split,
        name=args.name,
        data_dir=args.data_dir,
        revision=args.revision,
        hf_token=args.hf_token,
        min_chars=args.min_chars,
        log_every_docs=args.log_every_docs,
        workers=args.workers,
        source_workers=args.source_workers,
        source_reader=args.source_reader,
        max_docs=args.max_docs,
        validation_ratio=args.validation_ratio,
        split_seed=args.split_seed,
        encoding_name=args.encoding,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
