import math

import pytest
import torch

from model.config import ModelConfig
from model.gpt import GPT


def test_gpt_converts_token_sequences_into_vocabulary_logits() -> None:
    config = ModelConfig(
        vocab_size=32,
        d_model=16,
        n_layers=2,
        n_heads=4,
        max_seq_len=8,
        dropout=0.0,
    )
    model = GPT(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 6))

    logits, loss = model(input_ids)

    assert logits.shape == (2, 6, config.vocab_size)
    assert loss is None


def test_gpt_ties_input_and_output_embeddings_when_configured() -> None:
    config = ModelConfig(
        vocab_size=32,
        d_model=16,
        n_layers=1,
        n_heads=4,
        max_seq_len=8,
        tie_embeddings=True,
    )

    model = GPT(config)

    assert model.lm_head.weight is model.token_embedding.weight


def test_gpt_keeps_input_and_output_embeddings_independent_when_configured() -> None:
    config = ModelConfig(
        vocab_size=32,
        d_model=16,
        n_layers=1,
        n_heads=4,
        max_seq_len=8,
        tie_embeddings=False,
    )

    model = GPT(config)

    assert model.lm_head.weight is not model.token_embedding.weight


def test_gpt_training_batch_produces_finite_loss_and_gradients() -> None:
    torch.manual_seed(42)
    config = ModelConfig(
        vocab_size=16,
        d_model=16,
        n_layers=2,
        n_heads=4,
        max_seq_len=6,
        dropout=0.0,
    )
    model = GPT(config)
    input_ids = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5, 6],
        ]
    )
    targets = torch.tensor(
        [
            [1, 2, 3, 4, 5, 6],
            [2, 3, 4, 5, 6, 7],
        ]
    )

    _, loss = model(input_ids, targets)

    assert loss is not None
    assert loss.ndim == 0
    assert torch.isfinite(loss)

    loss.backward()

    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_gpt_initial_loss_is_close_to_uniform_vocabulary_baseline() -> None:
    torch.manual_seed(42)
    config = ModelConfig(
        vocab_size=32,
        d_model=16,
        n_layers=1,
        n_heads=4,
        max_seq_len=8,
        dropout=0.0,
    )
    model = GPT(config)
    input_ids = torch.randint(0, config.vocab_size, (4, 8))
    targets = torch.randint(0, config.vocab_size, (4, 8))

    _, loss = model(input_ids, targets)

    assert loss is not None
    assert abs(loss.item() - math.log(config.vocab_size)) < 0.5


def test_gpt_does_not_leak_future_tokens_into_earlier_logits() -> None:
    torch.manual_seed(42)
    config = ModelConfig(
        vocab_size=32,
        d_model=16,
        n_layers=2,
        n_heads=4,
        max_seq_len=6,
        dropout=0.0,
    )
    model = GPT(config)
    model.eval()
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    changed_input_ids = input_ids.clone()
    changed_input_ids[:, -1] = 7

    original_logits, _ = model(input_ids)
    changed_logits, _ = model(changed_input_ids)

    torch.testing.assert_close(
        original_logits[:, :-1],
        changed_logits[:, :-1],
    )


def test_gpt_rejects_targets_with_a_different_shape() -> None:
    config = ModelConfig(
        vocab_size=16,
        d_model=16,
        n_layers=1,
        n_heads=4,
        max_seq_len=6,
    )
    model = GPT(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 6))
    targets = torch.randint(0, config.vocab_size, (2, 5))

    with pytest.raises(ValueError, match="targets must have the same shape"):
        model(input_ids, targets)


def test_gpt_rejects_input_ids_without_batch_and_sequence_dimensions() -> None:
    config = ModelConfig(
        vocab_size=16,
        d_model=16,
        n_layers=1,
        n_heads=4,
        max_seq_len=6,
    )
    model = GPT(config)
    input_ids = torch.randint(0, config.vocab_size, (6,))

    with pytest.raises(ValueError, match=r"input_ids must have shape \[batch, seq_len\]"):
        model(input_ids)


def test_gpt_rejects_sequences_longer_than_its_context() -> None:
    config = ModelConfig(
        vocab_size=16,
        d_model=16,
        n_layers=1,
        n_heads=4,
        max_seq_len=6,
    )
    model = GPT(config)
    input_ids = torch.randint(0, config.vocab_size, (1, 7))

    with pytest.raises(ValueError, match="exceeds max_seq_len=6"):
        model(input_ids)


@pytest.mark.parametrize("shape", [(0, 6), (1, 0)])
def test_gpt_rejects_empty_batches_and_sequences(shape: tuple[int, int]) -> None:
    config = ModelConfig(
        vocab_size=16,
        d_model=16,
        n_layers=1,
        n_heads=4,
        max_seq_len=6,
    )
    model = GPT(config)
    input_ids = torch.empty(shape, dtype=torch.long)

    with pytest.raises(ValueError, match="batch and sequence dimensions must be non-empty"):
        model(input_ids)


def test_gpt_rejects_non_long_token_ids_and_targets() -> None:
    config = ModelConfig(
        vocab_size=16,
        d_model=16,
        n_layers=1,
        n_heads=4,
        max_seq_len=6,
    )
    model = GPT(config)
    long_ids = torch.randint(0, config.vocab_size, (1, 6))
    float_ids = long_ids.float()

    with pytest.raises(ValueError, match="input_ids must have dtype torch.long"):
        model(float_ids)

    with pytest.raises(ValueError, match="targets must have dtype torch.long"):
        model(long_ids, float_ids)
