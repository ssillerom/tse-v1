import pytest

from model.config import ModelConfig


@pytest.mark.parametrize(
    ("d_model", "n_heads", "message"),
    [
        (10, 4, "divisible"),
        (12, 4, "head_dim must be even"),
    ],
)
def test_model_config_rejects_dimensions_incompatible_with_attention_and_rope(
    d_model: int,
    n_heads: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        ModelConfig(d_model=d_model, n_heads=n_heads)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_layers": 1.5}, "n_layers must be a positive integer"),
        ({"n_layers": True}, "n_layers must be a positive integer"),
        ({"n_heads": 4.0}, "n_heads must be a positive integer"),
        ({"norm_eps": float("nan")}, "norm_eps must be finite and positive"),
        ({"tie_embeddings": "yes"}, "tie_embeddings must be a boolean"),
        ({"use_sdpa": "yes"}, "use_sdpa must be a boolean"),
    ],
)
def test_model_config_rejects_invalid_public_settings(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        ModelConfig(**kwargs)
