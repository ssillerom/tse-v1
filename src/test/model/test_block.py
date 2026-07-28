import torch

from model.block import TransformerBlock
from model.config import ModelConfig


def _tiny_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=32,
        d_model=16,
        n_layers=2,
        n_heads=4,
        max_seq_len=6,
        dropout=0.0,
    )


def test_transformer_block_preserves_residual_stream_shape() -> None:
    block = TransformerBlock(_tiny_config())
    inputs = torch.randn(2, 6, 16)

    outputs = block(inputs)

    assert outputs.shape == inputs.shape
    assert torch.isfinite(outputs).all()


def test_transformer_block_does_not_leak_future_information() -> None:
    torch.manual_seed(42)
    block = TransformerBlock(_tiny_config())
    block.eval()
    inputs = torch.randn(1, 6, 16)
    changed_inputs = inputs.clone()
    changed_inputs[:, -1] = torch.randn(16)

    outputs = block(inputs)
    changed_outputs = block(changed_inputs)

    torch.testing.assert_close(outputs[:, :-1], changed_outputs[:, :-1])


def test_transformer_block_produces_finite_gradients() -> None:
    torch.manual_seed(42)
    block = TransformerBlock(_tiny_config())
    inputs = torch.randn(2, 6, 16, requires_grad=True)

    loss = block(inputs).square().mean()
    loss.backward()

    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()
    for parameter in block.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
