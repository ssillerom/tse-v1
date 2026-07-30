"""Deterministic, resumable sampling across prepared pretraining sources."""

import hashlib
import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Protocol

import torch
from torch.utils.data import Dataset, Sampler

from src.data.recipe import RecipePhase

PretrainingSample = tuple[torch.Tensor, torch.Tensor]


class PretrainingSource(Protocol):
    """Sized random-access source accepted by the mixture."""

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> PretrainingSample: ...


class MixtureDataset(Dataset[PretrainingSample]):
    """Expose named datasets through one global integer index space."""

    def __init__(self, sources: Mapping[str, PretrainingSource]) -> None:
        if not sources:
            raise ValueError("sources must not be empty")
        self.source_names = tuple(sources)
        self.datasets = tuple(sources.values())
        self.source_lengths = tuple(len(dataset) for dataset in self.datasets)
        if any(length <= 0 for length in self.source_lengths):
            raise ValueError("every mixture source must contain at least one sequence")

        offsets: list[int] = []
        running_total = 0
        for length in self.source_lengths:
            offsets.append(running_total)
            running_total += length
        self.source_offsets = tuple(offsets)
        self.total_sequences = running_total
        self._source_positions = {name: position for position, name in enumerate(self.source_names)}

    def __len__(self) -> int:
        return self.total_sequences

    def __getitem__(self, index: int) -> PretrainingSample:
        if not isinstance(index, int) or index < 0 or index >= len(self):
            raise IndexError(f"Mixture index {index!r} is out of range")
        source_position = self._source_position_for_global_index(index)
        local_index = index - self.source_offsets[source_position]
        return self.datasets[source_position][local_index]

    def global_index(self, source_name: str, local_index: int) -> int:
        """Map one named source-local index into the combined index space."""
        if source_name not in self._source_positions:
            raise KeyError(f"Unknown mixture source {source_name!r}")
        source_position = self._source_positions[source_name]
        source_length = self.source_lengths[source_position]
        if not 0 <= local_index < source_length:
            raise IndexError(
                f"Local index {local_index} is out of range for source {source_name!r}"
            )
        return self.source_offsets[source_position] + local_index

    def source_length(self, source_name: str) -> int:
        if source_name not in self._source_positions:
            raise KeyError(f"Unknown mixture source {source_name!r}")
        return self.source_lengths[self._source_positions[source_name]]

    def source_for_global_index(self, index: int) -> str:
        if not isinstance(index, int) or index < 0 or index >= len(self):
            raise IndexError(f"Mixture index {index!r} is out of range")
        return self.source_names[self._source_position_for_global_index(index)]

    def _source_position_for_global_index(self, index: int) -> int:
        for position in range(len(self.source_offsets) - 1, -1, -1):
            if index >= self.source_offsets[position]:
                return position
        raise RuntimeError("failed to resolve mixture index")


@dataclass(frozen=True)
class _PhasePlan:
    name: str
    sequence_count: int
    source_ranges: tuple[tuple[str, int, int], ...]
    source_sequence_counts: dict[str, int]
    multiplier: int
    shift: int

    def assignment_at(self, position: int) -> tuple[str, int]:
        rank = (position * self.multiplier + self.shift) % self.sequence_count
        for source_name, range_start, range_end in self.source_ranges:
            if rank < range_end:
                return source_name, rank - range_start
        raise RuntimeError("failed to assign a phase position to a source")


def _stable_seed(seed: int, label: str) -> int:
    hasher = hashlib.blake2b(digest_size=8, person=b"llmfmix")
    hasher.update(str(seed).encode("ascii"))
    hasher.update(b"\0")
    hasher.update(label.encode("utf-8"))
    return int.from_bytes(hasher.digest(), byteorder="big")


def _coprime_multiplier(modulus: int, seed: int) -> int:
    if modulus <= 1:
        return 0
    candidate = seed % modulus
    if candidate == 0:
        candidate = 1
    while math.gcd(candidate, modulus) != 1:
        candidate = (candidate + 1) % modulus
        if candidate == 0:
            candidate = 1
    return candidate


def _phase_plan(phase: RecipePhase, seq_len: int, seed: int) -> _PhasePlan:
    if phase.target_tokens % seq_len != 0:
        raise ValueError(f"Phase {phase.name!r} tokens must be divisible by seq_len={seq_len}")
    if any(token_count % seq_len != 0 for _, token_count in phase.source_tokens):
        raise ValueError(
            f"Phase {phase.name!r} source token quotas must be divisible by seq_len={seq_len}"
        )

    source_ranges: list[tuple[str, int, int]] = []
    source_sequence_counts: dict[str, int] = {}
    range_start = 0
    for source_name, token_count in phase.source_tokens:
        sequence_count = token_count // seq_len
        range_end = range_start + sequence_count
        source_ranges.append((source_name, range_start, range_end))
        source_sequence_counts[source_name] = sequence_count
        range_start = range_end

    phase_seed = _stable_seed(seed, phase.name)
    total_sequences = phase.target_tokens // seq_len
    return _PhasePlan(
        name=phase.name,
        sequence_count=total_sequences,
        source_ranges=tuple(source_ranges),
        source_sequence_counts=source_sequence_counts,
        multiplier=_coprime_multiplier(total_sequences, phase_seed),
        shift=phase_seed % total_sequences,
    )


class DeterministicMixtureSampler(Sampler[int]):
    """Interleave sources and shuffle local blocks without materializing an index list."""

    def __init__(
        self,
        dataset: MixtureDataset,
        phases: tuple[RecipePhase, ...],
        seq_len: int,
        seed: int,
        start_position: int = 0,
        block_size: int = 4_096,
    ) -> None:
        if not isinstance(seq_len, int) or isinstance(seq_len, bool) or seq_len <= 0:
            raise ValueError("seq_len must be a positive integer")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("seed must be an integer")
        if (
            not isinstance(start_position, int)
            or isinstance(start_position, bool)
            or start_position < 0
        ):
            raise ValueError("start_position must be a non-negative integer")
        if not isinstance(block_size, int) or isinstance(block_size, bool) or block_size <= 0:
            raise ValueError("block_size must be a positive integer")
        if not phases:
            raise ValueError("phases must not be empty")

        self.dataset = dataset
        self.seed = seed
        self.block_size = block_size
        self.phase_plans = tuple(
            _phase_plan(phase, seq_len=seq_len, seed=seed + position)
            for position, phase in enumerate(phases)
        )
        unknown_sources = {
            source_name
            for plan in self.phase_plans
            for source_name in plan.source_sequence_counts
            if source_name not in dataset.source_names
        }
        if unknown_sources:
            raise ValueError(
                f"Recipe phases refer to unknown mixture sources: {sorted(unknown_sources)}"
            )
        self.total_sequences = sum(plan.sequence_count for plan in self.phase_plans)
        if start_position > self.total_sequences:
            raise ValueError(
                f"start_position cannot exceed {self.total_sequences}, got {start_position}"
            )
        self.start_position = start_position
        self._prior_source_counts = self._build_prior_source_counts()
        self._validate_capacities()

    def __len__(self) -> int:
        return self.total_sequences - self.start_position

    def __iter__(self) -> Iterator[int]:
        phase_start = 0
        for phase_position, plan in enumerate(self.phase_plans):
            phase_end = phase_start + plan.sequence_count
            local_start = max(self.start_position - phase_start, 0)
            if local_start < plan.sequence_count:
                for position in range(local_start, plan.sequence_count):
                    source_name, phase_source_occurrence = plan.assignment_at(position)
                    source_occurrence = (
                        self._prior_source_counts[phase_position][source_name]
                        + phase_source_occurrence
                    )
                    local_index = self._permuted_local_index(
                        source_name=source_name,
                        occurrence=source_occurrence,
                    )
                    yield self.dataset.global_index(source_name, local_index)
            phase_start = phase_end

    def source_sequence_counts(
        self,
        start_position: int,
        end_position: int,
    ) -> dict[str, int]:
        """Count exact source assignments in a half-open recipe position range."""
        for name, value in (
            ("start_position", start_position),
            ("end_position", end_position),
        ):
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"{name} must be an integer")
        if not 0 <= start_position <= end_position <= self.total_sequences:
            raise ValueError(
                f"source count range must satisfy 0 <= start <= end <= {self.total_sequences}"
            )

        counts = {source_name: 0 for source_name in self.dataset.source_names}
        phase_start = 0
        for plan in self.phase_plans:
            phase_end = phase_start + plan.sequence_count
            overlap_start = max(start_position, phase_start)
            overlap_end = min(end_position, phase_end)
            for global_position in range(overlap_start, overlap_end):
                source_name, _ = plan.assignment_at(global_position - phase_start)
                counts[source_name] += 1
            phase_start = phase_end
            if phase_start >= end_position:
                break
        return counts

    def _build_prior_source_counts(self) -> tuple[dict[str, int], ...]:
        running_counts = {source_name: 0 for source_name in self.dataset.source_names}
        prior_counts: list[dict[str, int]] = []
        for plan in self.phase_plans:
            prior_counts.append(dict(running_counts))
            for source_name, sequence_count in plan.source_sequence_counts.items():
                running_counts[source_name] += sequence_count
        return tuple(prior_counts)

    def _validate_capacities(self) -> None:
        required_by_source = {source_name: 0 for source_name in self.dataset.source_names}
        for plan in self.phase_plans:
            for source_name, sequence_count in plan.source_sequence_counts.items():
                required_by_source[source_name] += sequence_count
        for source_name in self.dataset.source_names:
            required = required_by_source[source_name]
            available = self.dataset.source_length(source_name)
            if required > available:
                raise ValueError(
                    f"Source {source_name!r} does not contain enough sequences: "
                    f"recipe needs {required}, prepared source has {available}"
                )

    def _permuted_local_index(self, source_name: str, occurrence: int) -> int:
        source_length = self.dataset.source_length(source_name)
        full_blocks = source_length // self.block_size
        full_block_sequences = full_blocks * self.block_size
        if occurrence >= full_block_sequences or full_blocks <= 1:
            return occurrence

        logical_block, offset = divmod(occurrence, self.block_size)
        source_seed = _stable_seed(self.seed, source_name)
        multiplier = _coprime_multiplier(full_blocks, source_seed)
        shift = (source_seed // max(full_blocks, 1)) % full_blocks
        physical_block = (logical_block * multiplier + shift) % full_blocks
        return physical_block * self.block_size + offset
