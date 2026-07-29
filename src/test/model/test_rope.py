import math

import pytest
import torch

from model.rope import RoPECache, apply_rope


def test_rope_leaves_position_zero_unchanged_and_preserves_norms() -> None:
    cache = RoPECache(d_k=4, max_seq_len=3)
    inputs = torch.randn(2, 3, 2, 4)

    rotated = apply_rope(inputs, cache.get_freqs(seq_len=3))

    torch.testing.assert_close(rotated[:, 0], inputs[:, 0])
    torch.testing.assert_close(
        torch.linalg.vector_norm(rotated, dim=-1),
        torch.linalg.vector_norm(inputs, dim=-1),
    )


def test_rope_rotates_each_even_odd_pair_by_its_position_angle() -> None:
    cache = RoPECache(d_k=2, max_seq_len=2)
    inputs = torch.tensor([[[[1.0, 0.0]], [[1.0, 0.0]]]])

    rotated = apply_rope(inputs, cache.get_freqs(seq_len=2))

    expected = torch.tensor([[[[1.0, 0.0]], [[math.cos(1.0), math.sin(1.0)]]]])
    torch.testing.assert_close(rotated, expected)


def test_rope_cache_follows_module_dtype() -> None:
    cache = RoPECache(d_k=4, max_seq_len=3).to(dtype=torch.float64)

    cos_f, sin_f = cache.get_freqs(seq_len=3)

    assert cos_f.dtype == torch.float64
    assert sin_f.dtype == torch.float64


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"d_k": 0, "max_seq_len": 8}, "d_k must be positive"),
        ({"d_k": 3, "max_seq_len": 8}, "d_k must be even"),
        ({"d_k": 4, "max_seq_len": 0}, "max_seq_len must be positive"),
        ({"d_k": 4, "max_seq_len": 8, "base": 0.0}, "base must be positive"),
    ],
)
def test_rope_cache_rejects_invalid_configuration(
    kwargs: dict[str, int | float],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        RoPECache(**kwargs)
