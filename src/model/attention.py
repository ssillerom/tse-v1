"""Multi-head causal self-attention: each token attends to its position and past.

The layer receives ``x`` with shape ``[batch, tokens, d_model]`` and creates
three learned representations:

- Query (Q): the information the current token is looking for.
- Key (K): how each token can be found by other tokens.
- Value (V): the information each token provides when it is attended to.

Q, K, and V are split into heads so that several relationship spaces can be
learned in parallel. RoPE rotates Q and K to inject position information.
Attention can use PyTorch SDPA, which selects an efficient kernel when
available, or the manual implementation retained for teaching. Both paths
apply scaling, the causal mask, softmax, and dropout before combining values.

The heads are then joined and ``W_o`` mixes their results. The output shape is
again ``[batch, tokens, d_model]`` so it can feed the Transformer's residual
connection. Different heads can learn different patterns, although their
projections are not required to remain disjoint.
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

        # Q determines what each token is looking for.
        self.W_q = nn.Linear(
            in_features=d_model,
            out_features=d_model,
            bias=qkv_bias,
        )

        # K describes how each token can be found.
        self.W_k = nn.Linear(
            in_features=d_model,
            out_features=d_model,
            bias=qkv_bias,
        )

        # V contains the information each token provides.
        self.W_v = nn.Linear(
            in_features=d_model,
            out_features=d_model,
            bias=qkv_bias,
        )

        # Mix the results produced by all heads.
        self.W_o = nn.Linear(
            in_features=d_model,
            out_features=d_model,
            bias=qkv_bias,
        )

        # Randomly drops attention weights after softmax during training.
        self.attention_dropout = nn.Dropout(dropout)

        causal_mask = None
        if not use_sdpa:
            # True marks future positions that the manual path must hide.
            causal_mask = torch.triu(
                torch.ones(
                    max_seq_len,
                    max_seq_len,
                    dtype=torch.bool,
                ),
                diagonal=1,
            )

        # This is not a trainable parameter, but it moves with the layer
        # when the layer is used on CPU, CUDA, or MPS.
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
        # 1. Create queries, keys, and values
        # ---------------------------------------------------------

        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        # Shape:
        # [batch_size, seq_len, d_model]

        # ---------------------------------------------------------
        # 2. Split d_model into multiple heads
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

        # Shape:
        # [batch_size, seq_len, n_heads, head_dim]

        # RoPE rotates queries and keys before their dot product so attention
        # scores encode the relative distance between token positions.
        freqs = self.rope_cache.get_freqs(seq_len)
        q = apply_rope(q, freqs)
        k = apply_rope(k, freqs)

        # ---------------------------------------------------------
        # 3. Move n_heads before seq_len
        # ---------------------------------------------------------

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Shape:
        # [batch_size, n_heads, seq_len, head_dim]

        # ---------------------------------------------------------
        # 4. Compute attention with SDPA or the manual path
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

        # Shape:
        # [batch_size, n_heads, seq_len, head_dim]

        # ---------------------------------------------------------
        # 5. Join the heads again
        # ---------------------------------------------------------

        context = context.transpose(1, 2)

        # Shape:
        # [batch_size, seq_len, n_heads, head_dim]

        context = context.contiguous().view(
            batch_size,
            seq_len,
            self.d_model,
        )

        # Shape:
        # [batch_size, seq_len, d_model]

        # ---------------------------------------------------------
        # 6. Mix information from all heads
        # ---------------------------------------------------------

        output: torch.Tensor = self.W_o(context)

        return output
