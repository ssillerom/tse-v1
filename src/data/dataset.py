"""Memory-mapped dataset for causal language-model pretraining."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


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

        # TODO 1: validar seq_len.
        self.seq_len = seq_len
        if not isinstance(seq_len, int) or isinstance(seq_len, bool) or seq_len <= 0:
            raise ValueError(f"seq_len must be a positive integer, got {seq_len!r}")

        # TODO 2: convertir manifest_path en Path y comprobar que existe.
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Manifest file not found: {self.manifest_path}")

        # TODO 3: cargar el JSON.
        self.manifest_data = self._load_manifest(self.manifest_path)

        # TODO 4: validar versión, dtype y existencia del split.
        format_version = self.manifest_data.get("format_version")
        if format_version != 2:
            raise ValueError(f"Unsupported manifest format_version: {format_version}")

        storage = self.manifest_data.get("storage")

        if not isinstance(storage, dict):
            raise ValueError("Manifest must contain a storage object")

        dtype = storage.get("dtype")

        if dtype != "uint16":
            raise ValueError(f"Unsupported storage dtype: {dtype!r}")

        splits = self.manifest_data.get("splits")

        if not isinstance(splits, dict):
            raise ValueError("Manifest must contain a splits object")

        if self.split not in splits:
            available_splits = ", ".join(sorted(splits))

            raise ValueError(
                f"Split {self.split!r} not found. Available splits: {available_splits}"
            )

        # TODO 5: filtrar los shards pertenecientes al split.
        raw_shards = self.manifest_data.get("shards")

        if not isinstance(raw_shards, list):
            raise ValueError("Manifest must contain a shards list")

        split_entries: list[dict[str, Any]] = []
        for position, entry in enumerate(raw_shards):
            if not isinstance(entry, dict):
                raise ValueError(f"Shard entry at position {position} is not an object")
            if entry.get("split") == self.split:
                split_entries.append(entry)

        if not split_entries:
            raise ValueError(f"No shards found for split {self.split!r}")

        self.shard_entries = split_entries

        # TODO 6: abrir y validar cada shard con _open_shard().
        self.shards = [
            self._open_shard(
                manifest_directory=self.manifest_path.parent,
                entry=entry,
            )
            for entry in split_entries
        ]

        # TODO 7: construir el índice acumulativo de secuencias.
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
        # TODO 8: comprobar que el índice está dentro del rango.
        if not isinstance(index, int) or index < 0 or index >= len(self):
            raise IndexError(f"Index {index} is out of range for dataset of length {len(self)}")

        # TODO 9: localizar shard y posición local.
        shard_idx, local_idx = self._locate_sequence(index)

        # TODO 10: extraer seq_len + 1 tokens.
        start = local_idx * self.seq_len
        tokens = self.shards[shard_idx].tokens[start : start + self.seq_len + 1]

        # TODO 11: convertir a torch.long.
        array = tokens.astype(np.int64, copy=True)

        torch_tokens = torch.from_numpy(array).to(torch.long)

        # TODO 12: separar input_ids y targets.
        input_ids = torch_tokens[: self.seq_len]
        target_ids = torch_tokens[1 : self.seq_len + 1]
        return input_ids, target_ids

    def _open_shard(
        self,
        manifest_directory: Path,
        entry: dict[str, Any],
    ) -> ShardIndex:
        """Resolve, validate and memory-map one manifest shard."""
        # TODO:
        # - Resolver entry["file"] respecto al directorio del manifest.
        # - Comprobar que el fichero existe.
        # - Leer entry["tokens"].
        # - Comprobar su tamaño en bytes.
        # - Abrirlo con np.memmap.
        # - Calcular sequence_count.
        # - Devolver ShardIndex.

        file_path = entry.get("file")

        if not isinstance(file_path, str) or not file_path:
            raise ValueError("Shard entry must contain a non-empty 'file' string")

        relative_path = Path(file_path)

        if relative_path.is_absolute():
            raise ValueError(f"Shard path must be relative: {file_path!r}")

        dataset_directory = manifest_directory.resolve()
        shard_path = (dataset_directory / relative_path).resolve()

        if not shard_path.is_relative_to(dataset_directory):
            raise ValueError(f"Shard path escapes the dataset directory: {file_path!r}")

        if not shard_path.is_file():
            raise FileNotFoundError(f"Shard file not found: {shard_path}")

        token_count = entry.get("tokens")

        if not isinstance(token_count, int) or isinstance(token_count, bool) or token_count <= 0:
            raise ValueError(f"Invalid token count for shard {shard_path}: {token_count!r}")

        bytes_per_token = np.dtype(np.uint16).itemsize
        expected_bytes = token_count * bytes_per_token
        actual_bytes = shard_path.stat().st_size

        if actual_bytes != expected_bytes:
            raise ValueError(
                f"Shard {shard_path} declares {token_count} tokens "
                f"and should contain {expected_bytes} bytes, "
                f"but contains {actual_bytes} bytes"
            )

        tokens = np.memmap(
            shard_path,
            dtype=np.uint16,
            mode="r",
        )

        sequence_count = (token_count - 1) // self.seq_len

        return ShardIndex(
            path=shard_path,
            token_count=token_count,
            sequence_count=sequence_count,
            tokens=tokens,
        )

    def _locate_sequence(self, index: int) -> tuple[int, int]:
        """Map a global sequence index to (shard_index, local_index)."""
        shard_index = int(np.searchsorted(self.cumulative_sequences, index, side="right"))
        previous_total = 0 if shard_index == 0 else int(self.cumulative_sequences[shard_index - 1])
        local_index = index - previous_total
        return shard_index, local_index

    @staticmethod
    def _load_manifest(path: Path) -> dict[str, Any]:
        """Load and minimally validate a prepared-data manifest."""

        text = path.read_text(encoding="utf-8")

        try:
            payload = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse manifest JSON: {e} in {path}") from e
        if not isinstance(payload, dict):
            raise ValueError(f"Manifest JSON of {path} must be an object")

        return payload
