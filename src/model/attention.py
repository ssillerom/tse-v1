"""Causal self-attention with optional grouped query heads and incremental K/V state.

The layer receives ``x`` with shape ``[batch, tokens, d_model]`` and creates
three learned representations:

- Query (Q): the information the current token is looking for.
- Key (K): how each token can be found by other tokens.
- Value (V): the information each token provides when it is attended to.

Q, K, and V are split into heads so that several relationship spaces can be
learned in parallel. With GQA, several query heads share each K/V head. RoPE
rotates Q and K to inject position information.
Attention can use PyTorch SDPA, which selects an efficient kernel when
available, or the manual implementation retained for teaching. Both paths
apply scaling, the causal mask, softmax, and dropout before combining values.

The heads are then joined and ``W_o`` mixes their results. The output shape is
again ``[batch, tokens, d_model]`` so it can feed the Transformer's residual
connection. Different heads can learn different patterns, although their
projections are not required to remain disjoint.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.rope import RoPECache, apply_rope


@dataclass(frozen=True)
class AttentionKVCache:
    """Compact rotated keys and values retained for one attention layer."""

    key: torch.Tensor
    value: torch.Tensor

    @property
    def seq_len(self) -> int:
        if self.key.ndim != 4 or self.value.ndim != 4:
            raise ValueError(
                "cache must have shape [batch_size, n_kv_heads, cached_tokens, head_dim]"
            )
        if self.key.shape != self.value.shape:
            raise ValueError("cached keys and values must have the same shape")
        return self.key.size(2)


class MultiHeadAttention(nn.Module):
    """
    Causal self-attention supporting MHA, GQA, and MQA configurations.

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
        n_kv_heads: int | None = None,
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

        if n_kv_heads is not None and (
            not isinstance(n_kv_heads, int) or isinstance(n_kv_heads, bool) or n_kv_heads <= 0
        ):
            raise ValueError(f"n_kv_heads must be a positive integer or None, got {n_kv_heads!r}")

        resolved_n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        if n_heads % resolved_n_kv_heads != 0:
            raise ValueError(f"n_heads ({n_heads}) must be divisible by n_kv_heads ({n_kv_heads})")

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = resolved_n_kv_heads
        self.n_repeat_kv = n_heads // self.n_kv_heads
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
            out_features=self.n_kv_heads * self.head_dim,
            bias=qkv_bias,
        )

        # V contains the information each token provides.
        self.W_v = nn.Linear(
            in_features=d_model,
            out_features=self.n_kv_heads * self.head_dim,
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
        output, _ = self._forward(x, cache=None, return_cache=False)
        return output

    def forward_with_cache(
        self,
        x: torch.Tensor,
        cache: AttentionKVCache | None = None,
    ) -> tuple[torch.Tensor, AttentionKVCache]:
        """Attend to new tokens and return compact K/V state for later decoding."""
        output, updated_cache = self._forward(x, cache=cache, return_cache=True)
        if updated_cache is None:
            raise RuntimeError("cached attention did not produce a cache")
        return output, updated_cache

    def _forward(
        self,
        x: torch.Tensor,
        *,
        cache: AttentionKVCache | None,
        return_cache: bool,
    ) -> tuple[torch.Tensor, AttentionKVCache | None]:
        if x.ndim != 3:
            raise ValueError("Expected x with shape [batch_size, seq_len, d_model]")

        batch_size, seq_len, input_dim = x.size()

        if input_dim != self.d_model:
            raise ValueError(f"Expected input dimension {self.d_model}, got {input_dim}")

        cached_seq_len = self._validate_cache(cache, x)
        total_seq_len = cached_seq_len + seq_len
        if total_seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {total_seq_len} exceeds max_seq_len={self.max_seq_len}"
            )

        # ---------------------------------------------------------
        # 1. Create queries, keys, and values
        # ---------------------------------------------------------

        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        if cache is not None and (cache.key.dtype != k.dtype or cache.value.dtype != v.dtype):
            raise ValueError("cache and projected keys/values must have the same dtype")

        # Q shape: [batch_size, seq_len, d_model]
        # K/V shape: [batch_size, seq_len, n_kv_heads * head_dim]

        # ---------------------------------------------------------
        # 2. Split projections into query heads and shared K/V heads
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
            self.n_kv_heads,
            self.head_dim,
        )

        v = v.view(
            batch_size,
            seq_len,
            self.n_kv_heads,
            self.head_dim,
        )

        # Q shape: [batch_size, seq_len, n_heads, head_dim]
        # K/V shape: [batch_size, seq_len, n_kv_heads, head_dim]

        # RoPE rotates queries and keys before their dot product so attention
        # scores encode the relative distance between token positions.
        all_freqs = self.rope_cache.get_freqs(total_seq_len)
        freqs = tuple(freq[cached_seq_len:total_seq_len] for freq in all_freqs)
        q = apply_rope(q, freqs)
        k = apply_rope(k, freqs)

        # ---------------------------------------------------------
        # 3. Move n_heads before seq_len
        # ---------------------------------------------------------

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if cache is not None:
            k = torch.cat((cache.key, k), dim=2)
            v = torch.cat((cache.value, v), dim=2)

        updated_cache = AttentionKVCache(key=k, value=v) if return_cache else None

        if self.n_repeat_kv > 1:
            k = k.repeat_interleave(self.n_repeat_kv, dim=1)
            v = v.repeat_interleave(self.n_repeat_kv, dim=1)

        # Shape:
        # [batch_size, n_heads, seq_len, head_dim]

        # ---------------------------------------------------------
        # 4. Compute attention with SDPA or the manual path
        # ---------------------------------------------------------

        allowed_attention_mask = None
        if cached_seq_len > 0:
            query_positions = torch.arange(
                cached_seq_len,
                total_seq_len,
                device=x.device,
            )
            key_positions = torch.arange(total_seq_len, device=x.device)
            allowed_attention_mask = query_positions.unsqueeze(1) >= key_positions.unsqueeze(0)

        if self.use_sdpa:
            context = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=allowed_attention_mask,
                dropout_p=self.attention_dropout.p if self.training else 0.0,
                is_causal=cached_seq_len == 0,
            )
        else:
            attention_scores = q @ k.transpose(-2, -1)
            attention_scores = attention_scores / math.sqrt(self.head_dim)

            if self.causal_mask is None:
                raise RuntimeError("Manual attention requires a causal mask")
            if cached_seq_len == 0:
                mask = self.causal_mask[:seq_len, :seq_len]
            else:
                if allowed_attention_mask is None:
                    raise RuntimeError("cached attention requires a causal mask")
                mask = ~allowed_attention_mask
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

        return output, updated_cache

    def _validate_cache(
        self,
        cache: AttentionKVCache | None,
        x: torch.Tensor,
    ) -> int:
        if cache is None:
            return 0
        cached_seq_len = cache.seq_len
        expected_prefix = (x.size(0), self.n_kv_heads)
        if cache.key.shape[:2] != expected_prefix:
            raise ValueError(
                "cache must have shape [batch_size, n_kv_heads, cached_tokens, head_dim]"
            )
        if cache.key.size(3) != self.head_dim:
            raise ValueError(f"cache head dimension must be {self.head_dim}")
        if cache.key.device != x.device or cache.value.device != x.device:
            raise ValueError("cache and input must be on the same device")
        return cached_seq_len
