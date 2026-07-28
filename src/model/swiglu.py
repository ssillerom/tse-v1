"""SwiGLU MLP: transforma las características de cada token por separado.

La atención mezcla información entre tokens; esta MLP no mezcla posiciones.
Recibe cada vector de tamaño ``d_model`` y crea dos proyecciones:

1. ``w_gate(x)`` pasa por SiLU y decide qué características dejar pasar.
2. ``w_up(x)`` contiene las características candidatas que se quieren procesar.
3. Ambas ramas se multiplican elemento a elemento y ``w_down`` devuelve el
   resultado a ``d_model`` para poder sumarlo a la conexión residual.

La operación es ``w_down(SiLU(w_gate(x)) * w_up(x))`` y conserva la forma
``[batch, tokens, d_model]``. La dimensión oculta usa aproximadamente
``8/3 * d_model`` para mantener un número de parámetros similar al de una MLP
clásica con expansión ``4 * d_model``, ya que SwiGLU tiene tres matrices.
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
