from collections import Counter

import pytest
import torch
from torch.utils.data import TensorDataset

from data.mixture import DeterministicMixtureSampler, MixtureDataset
from data.recipe import RecipePhase


def _mixture() -> MixtureDataset:
    return MixtureDataset(
        {
            "general": TensorDataset(torch.arange(100), torch.arange(100)),
            "code": TensorDataset(torch.arange(100, 140), torch.arange(100, 140)),
        }
    )


def _phases() -> tuple[RecipePhase, ...]:
    return (
        RecipePhase(
            name="stable",
            target_tokens=24,
            source_tokens=(("general", 16), ("code", 8)),
        ),
        RecipePhase(
            name="decay",
            target_tokens=8,
            source_tokens=(("general", 4), ("code", 4)),
        ),
    )


def test_mixture_sampler_is_deterministic_and_resumes_at_an_exact_position() -> None:
    mixture = _mixture()
    sampler = DeterministicMixtureSampler(
        dataset=mixture,
        phases=_phases(),
        seq_len=1,
        seed=42,
        block_size=4,
    )
    full_order = list(sampler)
    repeated_order = list(
        DeterministicMixtureSampler(
            dataset=mixture,
            phases=_phases(),
            seq_len=1,
            seed=42,
            block_size=4,
        )
    )
    resumed_order = list(
        DeterministicMixtureSampler(
            dataset=mixture,
            phases=_phases(),
            seq_len=1,
            seed=42,
            block_size=4,
            start_position=11,
        )
    )

    assert full_order == repeated_order
    assert resumed_order == full_order[11:]
    assert len(full_order) == 32


def test_mixture_sampler_follows_each_phase_source_ratio() -> None:
    mixture = _mixture()
    order = list(
        DeterministicMixtureSampler(
            dataset=mixture,
            phases=_phases(),
            seq_len=1,
            seed=7,
            block_size=4,
        )
    )
    source_names = [mixture.source_for_global_index(index) for index in order]

    assert Counter(source_names[:24]) == {"general": 16, "code": 8}
    assert Counter(source_names[24:]) == {"general": 4, "code": 4}


def test_mixture_sampler_rejects_a_recipe_larger_than_prepared_sources() -> None:
    mixture = MixtureDataset(
        {
            "general": TensorDataset(torch.arange(2), torch.arange(2)),
            "code": TensorDataset(torch.arange(2), torch.arange(2)),
        }
    )

    with pytest.raises(ValueError, match="does not contain enough sequences"):
        DeterministicMixtureSampler(
            dataset=mixture,
            phases=_phases(),
            seq_len=1,
            seed=42,
        )


def test_mixture_sampler_rejects_fractional_source_sequences() -> None:
    mixture = _mixture()
    phase = RecipePhase(
        name="stable",
        target_tokens=16,
        source_tokens=(("general", 10), ("code", 6)),
    )

    with pytest.raises(ValueError, match="source token quotas must be divisible"):
        DeterministicMixtureSampler(
            dataset=mixture,
            phases=(phase,),
            seq_len=4,
            seed=42,
        )
