import pytest
import torch

from model.rms_norm import RMSNorm


@pytest.mark.parametrize("input_dtype", [torch.float16, torch.bfloat16])
def test_rms_norm_preserves_low_precision_input_dtype_with_float32_scale(
    input_dtype: torch.dtype,
) -> None:
    norm = RMSNorm(d_model=4)
    inputs = torch.tensor(
        [[[1.0, -2.0, 3.0, -4.0]]],
        dtype=input_dtype,
    )

    output = norm(inputs)

    assert norm.gamma.dtype == torch.float32
    assert output.dtype == inputs.dtype
