"""Memory-mapped dataset for causal language-model pretraining."""

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.manifest import DataManifest, load_manifest


@dataclass(frozen=True)
class ShardIndex:
    """One validated shard and its contribution to the dataset."""

    path: Path
    token_count: int
    sequence_count: int

    @cached_property
    def tokens(self) -> np.memmap:
        """Open this process's read-only mapping on first use."""
        return np.memmap(self.path, dtype=np.uint16, mode="r")

    def __reduce__(self) -> tuple[type["ShardIndex"], tuple[Path, int, int]]:
        # Spawn workers receive metadata, never a serialized copy of token bytes.
        return type(self), (self.path, self.token_count, self.sequence_count)


class PretrainingDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Expose fixed-length causal-LM sequences from prepared token shards."""

    def __init__(
        self,
        manifest_path: str | Path | DataManifest,
        split: str,
        seq_len: int,
    ) -> None:
        self.split = split

        self.seq_len = seq_len
        if not isinstance(seq_len, int) or isinstance(seq_len, bool) or seq_len <= 0:
            raise ValueError(f"seq_len must be a positive integer, got {seq_len!r}")

        manifest = (
            manifest_path
            if isinstance(manifest_path, DataManifest)
            else load_manifest(manifest_path)
        )
        split_shards = manifest.shards_for_split(split)
        self.shards = [
            ShardIndex(shard.path, shard.token_count, (shard.token_count - 1) // seq_len)
            for shard in split_shards
        ]

        sequence_counts = np.array(
            [shard.sequence_count for shard in self.shards],
            dtype=np.int64,
        )

        self.cumulative_sequences = np.cumsum(sequence_counts)

        if self.cumulative_sequences.size == 0 or self.cumulative_sequences[-1] == 0:
            raise ValueError(
                f"Split {self.split!r} contains no complete sequences of length {self.seq_len}"
            )

    def __len__(self) -> int:
        """Return the total number of available sequences."""
        return int(self.cumulative_sequences[-1])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one input/target pair, both with shape [seq_len]."""
        if not isinstance(index, int) or index < 0 or index >= len(self):
            raise IndexError(f"Index {index} is out of range for dataset of length {len(self)}")

        shard_idx, local_idx = self._locate_sequence(index)

        start = local_idx * self.seq_len
        tokens = self.shards[shard_idx].tokens[start : start + self.seq_len + 1]

        array = tokens.astype(np.int64, copy=True)

        torch_tokens = torch.from_numpy(array)

        input_ids = torch_tokens[: self.seq_len]
        target_ids = torch_tokens[1 : self.seq_len + 1]
        return input_ids, target_ids

    def _locate_sequence(self, index: int) -> tuple[int, int]:
        """Map a global sequence index to (shard_index, local_index)."""
        shard_index = int(np.searchsorted(self.cumulative_sequences, index, side="right"))
        previous_total = 0 if shard_index == 0 else int(self.cumulative_sequences[shard_index - 1])
        local_index = index - previous_total
        return shard_index, local_index
