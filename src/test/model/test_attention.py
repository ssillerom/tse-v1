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
    )
    inputs = torch.randn(2, 6, 16, requires_grad=True)

    loss = attention(inputs).square().mean()
    loss.backward()

    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()
    for parameter in attention.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


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
