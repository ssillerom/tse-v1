"""Memory-mapped dataset for causal language-model pretraining."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.manifest import ManifestShard, load_manifest


@dataclass(frozen=True)
class ShardIndex:
    """One validated shard and its contribution to the dataset."""

    path: Path
    token_count: int
    sequence_count: int
    tokens: np.memmap


class PretrainingDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Expose fixed-length causal-LM sequences from prepared token shards."""

    def __init__(
        self,
        manifest_path: str | Path,
        split: str,
        seq_len: int,
    ) -> None:
        self.split = split

        # TODO 1: validate seq_len.
        self.seq_len = seq_len
        if not isinstance(seq_len, int) or isinstance(seq_len, bool) or seq_len <= 0:
            raise ValueError(f"seq_len must be a positive integer, got {seq_len!r}")

        manifest = load_manifest(manifest_path)
        split_shards = manifest.shards_for_split(split)
        self.shards = [self._open_shard(shard) for shard in split_shards]

        # TODO 7: build the cumulative sequence index.
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
        # TODO 8: check that the index is within bounds.
        if not isinstance(index, int) or index < 0 or index >= len(self):
            raise IndexError(f"Index {index} is out of range for dataset of length {len(self)}")

        # TODO 9: locate the shard and local position.
        shard_idx, local_idx = self._locate_sequence(index)

        # TODO 10: extract seq_len + 1 tokens.
        start = local_idx * self.seq_len
        tokens = self.shards[shard_idx].tokens[start : start + self.seq_len + 1]

        # TODO 11: convert to torch.long.
        array = tokens.astype(np.int64, copy=True)

        torch_tokens = torch.from_numpy(array).to(torch.long)

        # TODO 12: split input_ids and targets.
        input_ids = torch_tokens[: self.seq_len]
        target_ids = torch_tokens[1 : self.seq_len + 1]
        return input_ids, target_ids

    def _open_shard(
        self,
        shard: ManifestShard,
    ) -> ShardIndex:
        """Memory-map one already validated manifest shard."""
        tokens = np.memmap(
            shard.path,
            dtype=np.uint16,
            mode="r",
        )

        sequence_count = (shard.token_count - 1) // self.seq_len

        return ShardIndex(
            path=shard.path,
            token_count=shard.token_count,
            sequence_count=sequence_count,
            tokens=tokens,
        )

    def _locate_sequence(self, index: int) -> tuple[int, int]:
        """Map a global sequence index to (shard_index, local_index)."""
        shard_index = int(np.searchsorted(self.cumulative_sequences, index, side="right"))
        previous_total = 0 if shard_index == 0 else int(self.cumulative_sequences[shard_index - 1])
        local_index = index - previous_total
        return shard_index, local_index
