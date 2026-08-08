import pytest
import torch

from model.attention import MultiHeadAttention


def test_attention_preserves_batch_sequence_and_output_dimensions() -> None:
    attention = MultiHeadAttention(
        d_model=16,
        max_seq_len=8,
        dropout=0.0,
        n_heads=4,
    )
    inputs = torch.randn(2, 8, 16)

    context_vectors = attention(inputs)

    assert context_vectors.shape == (2, 8, 16)


def test_grouped_query_attention_reduces_key_value_parameters() -> None:
    mha = MultiHeadAttention(
        d_model=16,
        max_seq_len=8,
        dropout=0.0,
        n_heads=4,
    )
    gqa = MultiHeadAttention(
        d_model=16,
        max_seq_len=8,
        dropout=0.0,
        n_heads=4,
        n_kv_heads=2,
    )

    mha_parameter_count = sum(parameter.numel() for parameter in mha.parameters())
    gqa_parameter_count = sum(parameter.numel() for parameter in gqa.parameters())

    assert mha_parameter_count == 1_024
    assert gqa_parameter_count == 768


def test_attention_does_not_leak_information_from_future_tokens() -> None:

    torch.manual_seed(42)
    attention = MultiHeadAttention(
        d_model=16,
        max_seq_len=6,
        dropout=0.0,
        n_heads=4,
    )
    attention.eval()
    original_inputs = torch.randn(1, 6, 16)
    changed_inputs = original_inputs.clone()
    changed_inputs[:, -1] = torch.randn(16)

    original_output = attention(original_inputs)
    changed_output = attention(changed_inputs)

    torch.testing.assert_close(original_output[:, :-1], changed_output[:, :-1])


def test_grouped_query_attention_does_not_leak_information_from_future_tokens() -> None:
    torch.manual_seed(42)
    attention = MultiHeadAttention(
        d_model=16,
        max_seq_len=6,
        dropout=0.0,
        n_heads=4,
        n_kv_heads=2,
    )
    attention.eval()
    original_inputs = torch.randn(1, 6, 16)
    changed_inputs = original_inputs.clone()
    changed_inputs[:, -1] = torch.randn(16)

    original_output = attention(original_inputs)
    changed_output = attention(changed_inputs)

    torch.testing.assert_close(original_output[:, :-1], changed_output[:, :-1])


def test_attention_allows_past_tokens_to_influence_later_outputs() -> None:
    torch.manual_seed(42)
    attention = MultiHeadAttention(
        d_model=16,
        max_seq_len=6,
        dropout=0.0,
        n_heads=4,
    )
    attention.eval()
    original_inputs = torch.randn(1, 6, 16)
    changed_inputs = original_inputs.clone()
    changed_inputs[:, 0] += 10.0

    original_output = attention(original_inputs)
    changed_output = attention(changed_inputs)

    assert not torch.allclose(original_output[:, 1:], changed_output[:, 1:])


def test_attention_produces_finite_gradients() -> None:
    torch.manual_seed(42)
    attention = MultiHeadAttention(
        d_model=16,
        max_seq_len=6,
        dropout=0.0,
        n_heads=4,
        n_kv_heads=2,
    )
    inputs = torch.randn(2, 6, 16, requires_grad=True)

    loss = attention(inputs).square().mean()
    loss.backward()

    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()
    for parameter in attention.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_matching_query_and_key_value_head_counts_are_equivalent_to_mha() -> None:
    torch.manual_seed(42)
    mha = MultiHeadAttention(
        d_model=16,
        max_seq_len=6,
        dropout=0.0,
        n_heads=4,
    )
    explicit_mha = MultiHeadAttention(
        d_model=16,
        max_seq_len=6,
        dropout=0.0,
        n_heads=4,
        n_kv_heads=4,
    )
    explicit_mha.load_state_dict(mha.state_dict())
    inputs = torch.randn(2, 6, 16)

    torch.testing.assert_close(explicit_mha(inputs), mha(inputs))


def test_attention_accepts_sequences_shorter_than_its_context_length() -> None:
    attention = MultiHeadAttention(
        d_model=16,
        max_seq_len=8,
        dropout=0.0,
        n_heads=4,
    )
    inputs = torch.randn(2, 5, 16)

    context_vectors = attention(inputs)

    assert context_vectors.shape == (2, 5, 16)


def test_rope_theta_changes_attention_output_with_the_same_weights() -> None:
    torch.manual_seed(42)
    attention = MultiHeadAttention(
        d_model=16,
        n_heads=4,
        max_seq_len=8,
        dropout=0.0,
        rope_theta=10_000.0,
    )
    different_rope_theta = MultiHeadAttention(
        d_model=16,
        n_heads=4,
        max_seq_len=8,
        dropout=0.0,
        rope_theta=100.0,
    )
    different_rope_theta.load_state_dict(attention.state_dict())
    inputs = torch.randn(2, 8, 16)

    output = attention(inputs)
    output_with_different_rope_theta = different_rope_theta(inputs)

    assert not torch.allclose(output, output_with_different_rope_theta)


@pytest.mark.parametrize("use_sdpa", [False, True])
def test_grouped_query_attention_cache_matches_one_shot_attention(
    use_sdpa: bool,
) -> None:
    torch.manual_seed(42)
    attention = MultiHeadAttention(
        d_model=16,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=8,
        dropout=0.0,
        use_sdpa=use_sdpa,
    )
    attention.eval()
    inputs = torch.randn(2, 6, 16)

    expected = attention(inputs)
    prefix_output, cache = attention.forward_with_cache(inputs[:, :4])
    suffix_output, cache = attention.forward_with_cache(inputs[:, 4:], cache)

    torch.testing.assert_close(
        torch.cat((prefix_output, suffix_output), dim=1),
        expected,
        atol=1e-6,
        rtol=1e-5,
    )
    assert cache.key.shape == (2, 2, 6, 4)
    assert cache.value.shape == (2, 2, 6, 4)


def test_attention_cache_supports_cpu_mixed_precision_decoding() -> None:
    torch.manual_seed(42)
    attention = MultiHeadAttention(
        d_model=16,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=8,
        dropout=0.0,
    )
    attention.eval()
    inputs = torch.randn(1, 6, 16)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        _, cache = attention.forward_with_cache(inputs[:, :4])
        output, cache = attention.forward_with_cache(inputs[:, 4:], cache)

    assert output.shape == (1, 2, 16)
    assert cache.key.dtype == torch.bfloat16
