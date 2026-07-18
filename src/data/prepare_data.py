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
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import cast

import numpy as np
import tiktoken
from datasets import load_dataset  # type: ignore[import-untyped]

from .manifest import FORMAT_VERSION, STORAGE_DTYPE, write_manifest

LOGGER = logging.getLogger(__name__)
UINT16_MAX = int(np.iinfo(np.uint16).max)
SHARD_FILENAME_PATTERN = re.compile(r"shard_\d{4,}\.bin(?:\.tmp)?\Z")
GENERATED_DIRECTORY_NAMES = frozenset({"train", "validation"})


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

        self.shards.append(ShardMetadata(file=shard_path.name, tokens=token_count))
        LOGGER.info(
            "Saved shard %04d | %12s tokens | %14s total | %s",
            self.shard_idx,
            f"{token_count:,}",
            f"{self.total_tokens:,}",
            shard_path,
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
    destination_dir: Path,
    dataset: Iterable[Mapping[str, object]],
    dataset_manifest: Mapping[str, object],
    source_split: str,
    text_field: str,
    num_tokens: int,
    shard_size: int,
    min_chars: int,
    log_every_docs: int,
    max_docs: int | None,
    validation_ratio: float,
    split_seed: int,
    encoding: tiktoken.Encoding,
    encoding_name: str,
) -> dict[str, object]:
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
    LOGGER.info("Preparing dataset into %s", destination_dir)
    LOGGER.info(
        "Target: %s tokens | max documents: %s | shard size: %s | encoding: %s",
        f"{num_tokens:,}",
        "unlimited" if max_docs is None else f"{max_docs:,}",
        f"{shard_size:,}",
        encoding_name,
    )

    docs_seen = 0
    docs_used = 0
    docs_skipped = 0
    docs_truncated = 0
    total_tokens = 0
    started_at = time.perf_counter()

    for document in dataset:
        if max_docs is not None and docs_seen >= max_docs:
            break
        docs_seen += 1
        if text_field not in document:
            raise KeyError(
                f"text_field={text_field!r} not found. Available keys: {list(document.keys())}"
            )

        text = document[text_field]
        if not isinstance(text, str) or len(text) < min_chars:
            docs_skipped += 1
            continue

        tokens = encoding.encode(text, disallowed_special=())
        tokens.append(encoding.eot_token)
        output_split = _document_output_split(
            text,
            source_split,
            validation_ratio,
            split_seed,
        )
        writer = writers[output_split]
        remaining_tokens = num_tokens - total_tokens
        tokens_added = writer.add_tokens(tokens[:remaining_tokens])
        total_tokens += tokens_added
        if tokens_added > 0:
            docs_used += 1
            split_counts[output_split]["docs_used"] += 1
        if tokens_added < len(tokens):
            docs_truncated += 1
            split_counts[output_split]["docs_truncated"] += 1

        if docs_seen % log_every_docs == 0:
            elapsed = time.perf_counter() - started_at
            tokens_per_second = total_tokens / max(elapsed, 1e-9)
            LOGGER.info(
                "docs_seen=%s | docs_used=%s | skipped=%s | tokens=%s | tok/s=%s",
                f"{docs_seen:,}",
                f"{docs_used:,}",
                f"{docs_skipped:,}",
                f"{total_tokens:,}",
                f"{tokens_per_second:,.0f}",
            )

        if total_tokens >= num_tokens:
            break

    for writer in writers.values():
        writer.flush()
    elapsed = time.perf_counter() - started_at
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

    tokens_per_second = total_tokens / max(elapsed, 1e-9)
    LOGGER.info(
        "Done: %s tokens in %s shards from %s documents (%.2f h, %s tok/s)",
        f"{total_tokens:,}",
        f"{total_shards:,}",
        f"{docs_seen:,}",
        elapsed / 3600,
        f"{tokens_per_second:,.0f}",
    )
    return manifest


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
    max_docs: int | None = None,
    validation_ratio: float = 0.0,
    split_seed: int = 42,
    encoding_name: str = "gpt2",
    overwrite: bool = False,
) -> dict[str, object]:
    """Stream, tokenize, and transactionally replace a dataset and its manifest."""
    _validate_preparation_limits(
        num_tokens,
        shard_size,
        min_chars,
        log_every_docs,
        max_docs,
        validation_ratio,
    )

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

    dataset = load_streaming_hf_dataset(
        dataset_name=dataset_name,
        split=split,
        name=name,
        data_dir=data_dir,
        revision=revision,
        hf_token=hf_token,
    )
    dataset_manifest: dict[str, object] = {
        "path": dataset_name,
        "name": name,
        "data_dir": data_dir,
        "split": split,
        "revision": revision,
        "text_field": text_field,
    }

    with tempfile.TemporaryDirectory(
        prefix=".prepare-data-staging-",
        dir=output_path,
    ) as staging_name:
        staging_path = Path(staging_name)
        manifest = _prepare_into_staging_directory(
            staging_dir=staging_path,
            destination_dir=output_path,
            dataset=dataset,
            dataset_manifest=dataset_manifest,
            source_split=split,
            text_field=text_field,
            num_tokens=num_tokens,
            shard_size=shard_size,
            min_chars=min_chars,
            log_every_docs=log_every_docs,
            max_docs=max_docs,
            validation_ratio=validation_ratio,
            split_seed=split_seed,
            encoding=encoding,
            encoding_name=encoding_name,
        )
        _install_staged_artifacts(staging_path, output_path)

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
    prepare_parser.add_argument("--num-tokens", required=True, type=int)
    prepare_parser.add_argument("--shard-size", type=int, default=100_000_000)
    prepare_parser.add_argument("--min-chars", type=int, default=0)
    prepare_parser.add_argument("--log-every-docs", type=int, default=10_000)
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
    logging.basicConfig(level=logging.INFO, format="%(message)s")

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
        shard_size=args.shard_size,
        split=args.split,
        name=args.name,
        data_dir=args.data_dir,
        revision=args.revision,
        hf_token=args.hf_token,
        min_chars=args.min_chars,
        log_every_docs=args.log_every_docs,
        max_docs=args.max_docs,
        validation_ratio=args.validation_ratio,
        split_seed=args.split_seed,
        encoding_name=args.encoding,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
