"""Multi-head causal self-attention: cada token consulta su posición y el pasado.

La capa recibe ``x`` con forma ``[batch, tokens, d_model]`` y crea tres
representaciones aprendidas:

- Query (Q): qué información busca el token actual.
- Key (K): cómo puede ser encontrado cada token.
- Value (V): qué información entrega cada token si recibe atención.

Q, K y V se dividen en cabezas para que existan varios espacios de relación en
paralelo. RoPE rota Q y K para introducir posición. La atención puede calcularse
con PyTorch SDPA, que selecciona un kernel eficiente cuando está disponible, o
con la implementación manual conservada para aprendizaje. Ambas rutas aplican
el escalado, la máscara causal, softmax y dropout antes de combinar los values.

Finalmente se reúnen las cabezas y ``W_o`` mezcla sus resultados. La forma de
salida vuelve a ser ``[batch, tokens, d_model]`` para permitir la conexión
residual del Transformer. Las cabezas pueden aprender patrones distintos
gracias a sus proyecciones, aunque nada obliga a que no se solapen.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.rope import RoPECache, apply_rope


class MultiHeadAttention(nn.Module):
    """
    Standard causal multi-head self-attention.

    Input:
        x: [batch_size, seq_len, d_model]

    Output:
        [batch_size, seq_len, d_model]
    """

    causal_mask: torch.Tensor | None

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int,
        dropout: float = 0.0,
        qkv_bias: bool = False,
        rope_theta: float = 10_000.0,
        use_sdpa: bool = True,
    ) -> None:
        super().__init__()

        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")

        if n_heads <= 0:
            raise ValueError(f"n_heads must be positive, got {n_heads}")

        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")

        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")

        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        if rope_theta <= 0.0:
            raise ValueError(f"rope_theta must be positive, got {rope_theta}")

        if not isinstance(use_sdpa, bool):
            raise ValueError(f"use_sdpa must be a boolean, got {use_sdpa!r}")

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.max_seq_len = max_seq_len
        self.use_sdpa = use_sdpa

        if self.head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for RoPE, got {self.head_dim}")

        self.rope_cache = RoPECache(
            d_k=self.head_dim,
            max_seq_len=max_seq_len,
            base=rope_theta,
        )

        # Q decide qué busca cada token.
        self.W_q = nn.Linear(
            in_features=d_model,
            out_features=d_model,
            bias=qkv_bias,
        )

        # K describe cómo puede ser encontrado cada token.
        self.W_k = nn.Linear(
            in_features=d_model,
            out_features=d_model,
            bias=qkv_bias,
        )

        # V contiene la información que entrega cada token.
        self.W_v = nn.Linear(
            in_features=d_model,
            out_features=d_model,
            bias=qkv_bias,
        )

        # Mezcla los resultados producidos por todas las cabezas.
        self.W_o = nn.Linear(
            in_features=d_model,
            out_features=d_model,
            bias=qkv_bias,
        )

        # Randomly drops attention weights after softmax during training.
        self.attention_dropout = nn.Dropout(dropout)

        causal_mask = None
        if not use_sdpa:
            # True marca las posiciones futuras que debe ocultar la ruta manual.
            causal_mask = torch.triu(
                torch.ones(
                    max_seq_len,
                    max_seq_len,
                    dtype=torch.bool,
                ),
                diagonal=1,
            )

        # No es un parámetro entrenable, pero se moverá con la capa
        # cuando se utilice CPU, CUDA o MPS.
        self.register_buffer(
            "causal_mask",
            causal_mask,
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("Expected x with shape [batch_size, seq_len, d_model]")

        batch_size, seq_len, input_dim = x.size()

        if input_dim != self.d_model:
            raise ValueError(f"Expected input dimension {self.d_model}, got {input_dim}")

        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len={self.max_seq_len}")

        # ---------------------------------------------------------
        # 1. Crear queries, keys y values
        # ---------------------------------------------------------

        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        # Forma:
        # [batch_size, seq_len, d_model]

        # ---------------------------------------------------------
        # 2. Separar d_model en múltiples cabezas
        # ---------------------------------------------------------

        q = q.view(
            batch_size,
            seq_len,
            self.n_heads,
            self.head_dim,
        )

        k = k.view(
            batch_size,
            seq_len,
            self.n_heads,
            self.head_dim,
        )

        v = v.view(
            batch_size,
            seq_len,
            self.n_heads,
            self.head_dim,
        )

        # Forma:
        # [batch_size, seq_len, n_heads, head_dim]

        # RoPE rotates queries and keys before their dot product so attention
        # scores encode the relative distance between token positions.
        freqs = self.rope_cache.get_freqs(seq_len)
        q = apply_rope(q, freqs)
        k = apply_rope(k, freqs)

        # ---------------------------------------------------------
        # 3. Colocar n_heads antes de seq_len
        # ---------------------------------------------------------

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Forma:
        # [batch_size, n_heads, seq_len, head_dim]

        # ---------------------------------------------------------
        # 4. Calcular atención con SDPA o con la ruta manual
        # ---------------------------------------------------------

        if self.use_sdpa:
            context = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.attention_dropout.p if self.training else 0.0,
                is_causal=True,
            )
        else:
            attention_scores = q @ k.transpose(-2, -1)
            attention_scores = attention_scores / math.sqrt(self.head_dim)

            if self.causal_mask is None:
                raise RuntimeError("Manual attention requires a causal mask")
            mask = self.causal_mask[:seq_len, :seq_len]
            attention_scores = attention_scores.masked_fill(
                mask,
                -torch.inf,
            )

            attention_weights = torch.softmax(
                attention_scores,
                dim=-1,
            )
            attention_weights = self.attention_dropout(attention_weights)
            context = attention_weights @ v

        # Forma:
        # [batch_size, n_heads, seq_len, head_dim]

        # ---------------------------------------------------------
        # 5. Volver a unir las cabezas
        # ---------------------------------------------------------

        context = context.transpose(1, 2)

        # Forma:
        # [batch_size, seq_len, n_heads, head_dim]

        context = context.contiguous().view(
            batch_size,
            seq_len,
            self.d_model,
        )

        # Forma:
        # [batch_size, seq_len, d_model]

        # ---------------------------------------------------------
        # 6. Mezclar la información de todas las cabezas
        # ---------------------------------------------------------

        output: torch.Tensor = self.W_o(context)

        return output
