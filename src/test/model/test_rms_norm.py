import pytest
import torch

from model.rms_norm import RMSNorm


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"d_model": 0}, "d_model.*positive"),
        ({"d_model": 8, "eps": 0.0}, "eps.*positive"),
    ],
)
def test_rms_norm_rejects_invalid_configuration(
    kwargs: dict[str, int | float],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        RMSNorm(**kwargs)


def test_rms_norm_matches_pytorch_reference_and_preserves_shape() -> None:
    norm = RMSNorm(d_model=4, eps=1e-6)
    reference = torch.nn.RMSNorm(4, eps=1e-6)
    inputs = torch.tensor(
        [
            [[1.0, -2.0, 3.0, -4.0], [4.0, 3.0, 2.0, 1.0]],
            [[0.5, -0.5, 1.5, -1.5], [2.0, -1.0, 0.0, 3.0]],
        ]
    )

    with torch.no_grad():
        norm.gamma.copy_(torch.tensor([0.5, 1.0, 1.5, 2.0]))
        reference.weight.copy_(norm.gamma)

    output = norm(inputs)
    expected = reference(inputs)

    assert output.shape == inputs.shape
    torch.testing.assert_close(output, expected)


def test_rms_norm_produces_finite_gradients_for_input_and_scale() -> None:
    torch.manual_seed(42)
    norm = RMSNorm(d_model=8)
    inputs = torch.randn(2, 4, 8, requires_grad=True)

    loss = norm(inputs).square().mean()
    loss.backward()

    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()
    assert norm.gamma.grad is not None
    assert torch.isfinite(norm.gamma.grad).all()


def test_rms_norm_keeps_zero_inputs_finite() -> None:
    norm = RMSNorm(d_model=4)

    output = norm(torch.zeros(2, 3, 4))

    assert torch.isfinite(output).all()
    torch.testing.assert_close(output, torch.zeros_like(output))


def test_rms_norm_avoids_overflow_for_large_float16_inputs() -> None:
    norm = RMSNorm(d_model=4).to(dtype=torch.float16)
    inputs = torch.full((1, 2, 4), 60_000.0, dtype=torch.float16)

    output = norm(inputs)

    torch.testing.assert_close(
        output,
        torch.ones_like(output),
        atol=1e-3,
        rtol=1e-3,
    )
