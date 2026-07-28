"""RMSNorm: estabiliza la escala del vector que representa cada token.

Para cada token calcula la raíz de la media de los cuadrados de sus
características y divide el vector entre ese valor:
``x_normalizado = x / sqrt(mean(x**2) + eps)``.

Esto mantiene controlada la magnitud de las activaciones sin restar su media,
a diferencia de LayerNorm. Después, el parámetro entrenable ``gamma`` permite
que el modelo recupere o ajuste la escala útil de cada característica. ``eps``
evita divisiones inestables y el cálculo se hace en float32 para proteger
entradas float16/bfloat16 frente a desbordamientos. La forma del tensor no
cambia: ``[batch, tokens, d_model]`` entra y sale.
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
