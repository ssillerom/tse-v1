"""SwiGLU MLP: transform each token's features independently.

Attention mixes information between tokens; this MLP does not mix positions.
It receives each vector of size ``d_model`` and creates two projections:

1. ``w_gate(x)`` passes through SiLU and chooses which features to allow through.
2. ``w_up(x)`` contains the candidate features to process.
3. The branches are multiplied elementwise and ``w_down`` returns the result to
   ``d_model`` so it can be added to the residual connection.

The operation is ``w_down(SiLU(w_gate(x)) * w_up(x))`` and preserves the shape
``[batch, tokens, d_model]``. The hidden dimension uses approximately
``8/3 * d_model`` to keep the parameter count close to a classic MLP with a
``4 * d_model`` expansion, because SwiGLU uses three matrices.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """
    SwiGLU feed forward network. Uses the now common SwiGLU architecture.
    Has 3 learnable weight matrices: SwiGLU(x) = (SiLU(xWgate) * xWup) Wdown.
    Uses an 8/3 expansion factor to preserve approximately the parameter count
    of a classic MLP with a 4 * d_model hidden dimension.
    """

    def __init__(self, d_model: int, bias: bool = False):
        super().__init__()

        if not isinstance(d_model, int) or isinstance(d_model, bool) or d_model <= 0:
            raise ValueError(f"d_model must be a positive integer, got {d_model!r}")

        self.d_model = d_model
        hidden_dim = int(8 / 3 * d_model)
        self.w_gate = nn.Linear(d_model, hidden_dim, bias=bias)
        self.w_up = nn.Linear(d_model, hidden_dim, bias=bias)
        self.w_down = nn.Linear(hidden_dim, d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))
