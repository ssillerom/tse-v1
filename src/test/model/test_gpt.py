import math

import pytest
import torch

from model.attention import AttentionKVCache
from model.config import ModelConfig
from model.gpt import GPT, GPTKVCache


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize(
    ("input_ids", "message"),
    [
        (torch.ones(3, dtype=torch.long), "shape"),
        (torch.ones(1, 3), "dtype"),
        (torch.empty(0, 3, dtype=torch.long), "non-empty"),
        (torch.empty(1, 0, dtype=torch.long), "non-empty"),
        (torch.ones(1, 5, dtype=torch.long), "exceeds"),
    ],
)
def test_cached_and_uncached_forward_validate_inputs_identically(input_ids, message, cached):
    model = GPT(ModelConfig(vocab_size=8, d_model=8, n_layers=1, n_heads=2, max_seq_len=4))
    forward = model.forward_with_cache if cached else model.forward
    with pytest.raises(ValueError, match=message):
        forward(input_ids)


def test_cached_forward_counts_prefix_against_context_limit():
    model = GPT(ModelConfig(vocab_size=8, d_model=8, n_layers=1, n_heads=2, max_seq_len=4))
    _, cache = model.forward_with_cache(torch.ones(1, 4, dtype=torch.long))
    with pytest.raises(ValueError, match="Sequence length 5 exceeds"):
        model.forward_with_cache(torch.ones(1, 1, dtype=torch.long), cache)


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


def test_gpt_propagates_grouped_query_attention_to_every_block() -> None:
    base_settings = {
        "vocab_size": 32,
        "d_model": 16,
        "n_layers": 2,
        "n_heads": 4,
        "max_seq_len": 8,
        "dropout": 0.0,
    }
    mha = GPT(ModelConfig(**base_settings))
    gqa = GPT(ModelConfig(**base_settings, n_kv_heads=2))

    mha_parameter_count = sum(parameter.numel() for parameter in mha.parameters())
    gqa_parameter_count = sum(parameter.numel() for parameter in gqa.parameters())

    assert mha_parameter_count - gqa_parameter_count == 512


def test_gpt_can_use_manual_attention_without_calling_sdpa(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable_sdpa(*args: object, **kwargs: object) -> torch.Tensor:
        raise RuntimeError("SDPA is unavailable")

    monkeypatch.setattr(
        torch.nn.functional,
        "scaled_dot_product_attention",
        unavailable_sdpa,
    )
    config = ModelConfig(
        vocab_size=32,
        d_model=16,
        n_layers=2,
        n_heads=4,
        max_seq_len=8,
        dropout=0.0,
        use_sdpa=False,
    )
    model = GPT(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 6))

    logits, _ = model(input_ids)

    assert logits.shape == (2, 6, config.vocab_size)


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


@pytest.mark.parametrize("use_sdpa", [False, True])
def test_gpt_kv_cache_matches_one_shot_logits(use_sdpa: bool) -> None:
    torch.manual_seed(42)
    config = ModelConfig(
        vocab_size=32,
        d_model=16,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=8,
        dropout=0.0,
        use_sdpa=use_sdpa,
    )
    model = GPT(config)
    model.eval()
    input_ids = torch.randint(0, config.vocab_size, (2, 6))

    expected_logits, _ = model(input_ids)
    prefix_logits, cache = model.forward_with_cache(input_ids[:, :4])
    suffix_logits, cache = model.forward_with_cache(input_ids[:, 4:], cache)

    torch.testing.assert_close(
        torch.cat((prefix_logits, suffix_logits), dim=1),
        expected_logits,
        atol=1e-5,
        rtol=1e-5,
    )
    assert cache.seq_len == 6
    assert len(cache.layers) == config.n_layers


def test_gpt_rejects_malformed_kv_cache_with_an_explicit_error() -> None:
    config = ModelConfig(
        vocab_size=16,
        d_model=16,
        n_layers=1,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=8,
    )
    model = GPT(config)
    malformed_tensor = torch.zeros(1, 2)
    malformed_cache = GPTKVCache(
        (AttentionKVCache(key=malformed_tensor, value=malformed_tensor.clone()),)
    )

    with pytest.raises(ValueError, match="cache must have shape"):
        model.forward_with_cache(torch.tensor([[1]], dtype=torch.long), malformed_cache)
