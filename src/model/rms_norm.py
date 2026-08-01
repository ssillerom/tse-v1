"""RMSNorm: stabilize the scale of the vector representing each token.

For each token, RMSNorm computes the root mean square of its features and
divides the vector by that value:
``x_normalized = x / sqrt(mean(x**2) + eps)``.

This controls activation magnitude without subtracting the mean, unlike
LayerNorm. The learned ``gamma`` parameter can then restore or adjust the
useful scale of each feature. ``eps`` prevents unstable divisions, and the
calculation uses float32 to protect float16/bfloat16 inputs from overflow. The
tensor shape is unchanged: ``[batch, tokens, d_model]`` enters and leaves.
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()

        if d_model <= 0:
            raise ValueError("d_model must be a positive integer")

        if eps <= 0:
            raise ValueError("eps must be a positive float")

        self.gamma = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        values = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        rms = torch.sqrt(torch.mean(values.square(), dim=-1, keepdim=True) + self.eps)
        normalized = values / rms
        return (normalized * self.gamma).to(dtype=input_dtype)
