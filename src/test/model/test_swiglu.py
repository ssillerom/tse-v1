import pytest
import torch

from model.swiglu import SwiGLU


@pytest.mark.parametrize("d_model", [0, -1, True])
def test_swiglu_rejects_invalid_model_dimension(d_model: int) -> None:
    with pytest.raises(ValueError, match="d_model.*positive"):
        SwiGLU(d_model=d_model)


def test_swiglu_preserves_batch_sequence_and_model_dimensions() -> None:
    swiglu = SwiGLU(d_model=8)
    inputs = torch.randn(2, 5, 8)

    output = swiglu(inputs)

    assert output.shape == inputs.shape
    assert torch.isfinite(output).all()


def test_swiglu_maps_zero_input_to_zero_without_bias() -> None:
    swiglu = SwiGLU(d_model=8, bias=False)
    inputs = torch.zeros(2, 5, 8)

    output = swiglu(inputs)

    torch.testing.assert_close(output, torch.zeros_like(output))


def test_swiglu_produces_finite_gradients_for_input_and_parameters() -> None:
    torch.manual_seed(42)
    swiglu = SwiGLU(d_model=8)
    inputs = torch.randn(2, 5, 8, requires_grad=True)

    loss = swiglu(inputs).square().mean()
    loss.backward()

    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()
    for parameter in swiglu.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
