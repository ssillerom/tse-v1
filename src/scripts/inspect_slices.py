"""Inspect pretraining dataset slices as token IDs and decoded text."""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import tiktoken

from src.data.dataset import PretrainingDataset


def _manifest_encoding_name(manifest_path: Path) -> str:
    payload: Any = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Manifest root must be a JSON object")

    tokenizer = payload.get("tokenizer")
    if not isinstance(tokenizer, dict):
        raise ValueError("Manifest must contain a tokenizer object")

    encoding_name = tokenizer.get("encoding")
    if not isinstance(encoding_name, str) or not encoding_name:
        raise ValueError("Manifest tokenizer must contain a non-empty encoding")
    return encoding_name


def inspect_slices(
    manifest_path: str | Path,
    split: str,
    seq_len: int,
    num_examples: int = 3,
    start_index: int = 0,
) -> None:
    """Print decoded causal-LM windows from a prepared dataset."""
    if num_examples <= 0:
        raise ValueError("num_examples must be positive")
    if start_index < 0:
        raise ValueError("start_index cannot be negative")

    path = Path(manifest_path)
    encoding_name = _manifest_encoding_name(path)
    encoding = tiktoken.get_encoding(encoding_name)
    dataset = PretrainingDataset(path, split=split, seq_len=seq_len)

    if start_index >= len(dataset):
        raise IndexError(
            f"start_index {start_index} is out of range for dataset of length {len(dataset)}"
        )

    stop_index = min(start_index + num_examples, len(dataset))
    print(f"Manifest: {path}")
    print(f"Tokenizer: {encoding_name}")
    print(f"Split: {split}")
    print(f"Sequence length: {seq_len}")
    print(f"Available sequences: {len(dataset):,}")
    print("=" * 100)

    previous_target_last: int | None = None
    for index in range(start_index, stop_index):
        input_ids, targets = dataset[index]
        input_token_ids = input_ids.tolist()
        target_token_ids = targets.tolist()
        window_token_ids = [*input_token_ids, target_token_ids[-1]]

        print(f"\nSEQUENCE {index}")
        print("-" * 100)
        print(f"WINDOW:\n{encoding.decode(window_token_ids)!r}")
        print(f"\nINPUT:\n{encoding.decode(input_token_ids)!r}")
        print(f"\nTARGET:\n{encoding.decode(target_token_ids)!r}")
        print(f"\nINPUT IDS:  {input_token_ids}")
        print(f"TARGET IDS: {target_token_ids}")
        print(f"SHIFT VALID: {input_token_ids[1:] == target_token_ids[:-1]}")

        if previous_target_last is not None:
            print(f"CONTINUES PREVIOUS: {previous_target_last == input_token_ids[0]}")
        previous_target_last = target_token_ids[-1]


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--split", default="train")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--num-examples", type=int, default=3)
    parser.add_argument("--start-index", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the slice inspection CLI."""
    args = build_parser().parse_args(argv)
    inspect_slices(
        manifest_path=args.manifest,
        split=args.split,
        seq_len=args.seq_len,
        num_examples=args.num_examples,
        start_index=args.start_index,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
