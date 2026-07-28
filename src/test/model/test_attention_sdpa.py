import pytest
import torch

from model.attention import MultiHeadAttention


def test_sdpa_does_not_allocate_a_quadratic_causal_mask() -> None:
    attention = MultiHeadAttention(
        d_model=16,
        n_heads=4,
        max_seq_len=1_024,
        use_sdpa=True,
    )

    assert attention.causal_mask is None


def test_sdpa_matches_manual_attention_with_the_same_weights() -> None:
    torch.manual_seed(42)
    settings = {
        "d_model": 16,
        "n_heads": 4,
        "max_seq_len": 6,
        "dropout": 0.0,
    }
    manual_attention = MultiHeadAttention(**settings, use_sdpa=False)
    sdpa_attention = MultiHeadAttention(**settings, use_sdpa=True)
    sdpa_attention.load_state_dict(manual_attention.state_dict())
    inputs = torch.randn(2, 6, 16)

    manual_output = manual_attention(inputs)
    sdpa_output = sdpa_attention(inputs)

    torch.testing.assert_close(sdpa_output, manual_output, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("use_sdpa", [False, True])
def test_attention_dropout_follows_the_module_training_mode(use_sdpa: bool) -> None:
    torch.manual_seed(42)
    attention = MultiHeadAttention(
        d_model=16,
        n_heads=4,
        max_seq_len=6,
        dropout=0.5,
        use_sdpa=use_sdpa,
    )
    inputs = torch.randn(2, 6, 16)

    attention.train()
    torch.manual_seed(1)
    first_training_output = attention(inputs)
    torch.manual_seed(2)
    second_training_output = attention(inputs)

    assert not torch.allclose(first_training_output, second_training_output)

    attention.eval()
    first_evaluation_output = attention(inputs)
    second_evaluation_output = attention(inputs)

    torch.testing.assert_close(first_evaluation_output, second_evaluation_output)
