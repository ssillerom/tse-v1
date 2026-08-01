"""Evaluate one GPT checkpoint and publish a reproducible benchmark artifact.

The script reconstructs the raw model from its checkpoint, adapts it to the
EleutherAI harness, then atomically writes results together with enough model,
tokenizer, dependency, and Git metadata to identify the exact evaluation.

Author: Sergio Sillero.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import tiktoken
import torch

from src.evaluation.harness_adapter import (
    LM_EVAL_AVAILABLE,
    GPT2HarnessAdapter,
    load_evaluation_checkpoint,
)

DEFAULT_BASE_TASKS = (
    "lambada_openai",
    "hellaswag",
    "arc_easy",
    "arc_challenge",
    "winogrande",
    "piqa",
    "wikitext",
)


@dataclass(frozen=True)
class HarnessRuntime:
    """Optional harness boundary loaded only for a real benchmark run."""

    simple_evaluate: Callable[..., object]
    handle_non_serializable: Callable[[object], object]
    version: str


def _load_harness_runtime() -> HarnessRuntime:
    if not LM_EVAL_AVAILABLE:
        raise RuntimeError(
            "lm-evaluation-harness is optional; install it with `uv sync --extra eval`"
        )

    import lm_eval
    from lm_eval.utils import handle_non_serializable

    return HarnessRuntime(
        simple_evaluate=lm_eval.simple_evaluate,
        handle_non_serializable=handle_non_serializable,
        version=importlib.metadata.version("lm-eval"),
    )


def _parse_limit(value: str) -> int | float:
    try:
        parsed: int | float = float(value) if "." in value else int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("limit must be an integer or float") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("limit must be positive")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--tasks",
        default=",".join(DEFAULT_BASE_TASKS),
        help="comma-separated lm-evaluation-harness task names",
    )
    parser.add_argument("--num-fewshot", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=_parse_limit)
    parser.add_argument("--tokenizer", default="gpt2")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument(
        "--precision",
        choices=("auto", "fp32", "bf16"),
        default="auto",
    )
    parser.add_argument("--log-samples", action="store_true")
    return parser


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is not available")
    return torch.device(requested)


def _resolve_precision(requested: str, device: torch.device) -> str:
    if requested == "auto":
        return "bf16" if device.type == "cuda" and torch.cuda.is_bf16_supported() else "fp32"
    if requested == "bf16" and device.type == "mps":
        raise ValueError("bf16 evaluation is not supported on MPS")
    return requested


def _git_metadata() -> dict[str, object]:
    repository_root = Path(__file__).resolve().parents[2]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
        cwd=repository_root,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        check=False,
        capture_output=True,
        text=True,
        cwd=repository_root,
    )
    return {
        "commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "dirty": status.returncode != 0 or bool(status.stdout.strip()),
    }


def _checkpoint_sha256(path: Path) -> str:
    with path.open("rb") as checkpoint_file:
        return hashlib.file_digest(checkpoint_file, "sha256").hexdigest()


def main(
    argv: list[str] | None = None,
    *,
    harness_runtime: HarnessRuntime | None = None,
) -> int:
    """Run post-hoc benchmark evaluation and atomically save a JSON artifact."""
    arguments = _build_parser().parse_args(argv)
    if arguments.num_fewshot < 0:
        raise ValueError("num_fewshot must be non-negative")
    if arguments.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    tasks = tuple(task.strip() for task in arguments.tasks.split(",") if task.strip())
    if not tasks:
        raise ValueError("tasks must contain at least one task name")
    runtime = _load_harness_runtime() if harness_runtime is None else harness_runtime

    device = _resolve_device(arguments.device)
    precision = _resolve_precision(arguments.precision, device)
    encoding = tiktoken.get_encoding(arguments.tokenizer)
    checkpoint = load_evaluation_checkpoint(arguments.checkpoint, device=device)
    adapter = GPT2HarnessAdapter(
        model=checkpoint.model,
        encoding=encoding,
        device=device,
        batch_size=arguments.batch_size,
        precision=precision,
    )
    # The harness owns task prompts and metric definitions; the adapter owns
    # only tokenization, model scoring, and greedy generation.
    raw_results = runtime.simple_evaluate(
        model=adapter,
        tasks=list(tasks),
        num_fewshot=arguments.num_fewshot,
        batch_size=arguments.batch_size,
        limit=arguments.limit,
        log_samples=arguments.log_samples,
    )
    if not isinstance(raw_results, dict):
        raise RuntimeError("lm-evaluation-harness returned no result dictionary")

    # Keep provenance beside the raw harness payload rather than depending on
    # a mutable W&B run or the checkpoint filename alone.
    artifact: dict[str, Any] = {
        "format_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "checkpoint": {
            "path": str(checkpoint.path),
            "sha256": _checkpoint_sha256(checkpoint.path),
            "step": checkpoint.step,
        },
        "model_config": asdict(checkpoint.model.config),
        "evaluation": {
            "tasks": list(tasks),
            "num_fewshot": arguments.num_fewshot,
            "batch_size": arguments.batch_size,
            "limit": arguments.limit,
            "device": str(device),
            "precision": precision,
            "tokenizer": {
                "encoding": encoding.name,
                "vocab_size": encoding.n_vocab,
                "eot_token_id": encoding.eot_token,
            },
            "log_samples": arguments.log_samples,
        },
        "versions": {
            "torch": torch.__version__,
            "lm_eval": runtime.version,
        },
        "repository": _git_metadata(),
        "harness": raw_results,
    }
    serialized = json.dumps(
        artifact,
        indent=2,
        sort_keys=True,
        default=runtime.handle_non_serializable,
    )
    # Publish only a complete JSON document. A crash can leave the previous
    # result intact, never a partially-written artifact.
    output_path = cast(Path, arguments.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f"{output_path.name}.tmp")
    try:
        temporary_path.write_text(f"{serialized}\n", encoding="utf-8")
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
