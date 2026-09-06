"""V1 decoder-only language model.

GPT assembles the model's components into one causal path:
token IDs -> embeddings -> transformer blocks -> final RMSNorm -> logits.
RoPE lives inside each attention layer, so this module does not add separate
positional embeddings.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.attention import AttentionKVCache
from src.model.block import TransformerBlock
from src.model.config import ModelConfig
from src.model.rms_norm import RMSNorm

IGNORE_INDEX = -100


@dataclass(frozen=True)
class GPTKVCache:
    """Per-layer compact K/V state for incremental generation."""

    layers: tuple[AttentionKVCache, ...]

    @property
    def seq_len(self) -> int:
        if not self.layers:
            return 0
        lengths = {layer.seq_len for layer in self.layers}
        if len(lengths) != 1:
            raise ValueError("all layer caches must contain the same number of tokens")
        return lengths.pop()

    @classmethod
    def batch(cls, caches: Sequence["GPTKVCache"]) -> "GPTKVCache":
        """Join equal-length caches along their batch dimension."""
        if not caches:
            raise ValueError("at least one cache is required")
        layer_count = len(caches[0].layers)
        if layer_count == 0:
            raise ValueError("cache must contain at least one layer")
        expected_seq_len = caches[0].seq_len
        if any(
            len(cache.layers) != layer_count or cache.seq_len != expected_seq_len
            for cache in caches
        ):
            raise ValueError("all caches must have matching layers and sequence lengths")
        return cls(
            tuple(
                AttentionKVCache(
                    key=torch.cat(
                        [cache.layers[layer_index].key for cache in caches],
                        dim=0,
                    ),
                    value=torch.cat(
                        [cache.layers[layer_index].value for cache in caches],
                        dim=0,
                    ),
                )
                for layer_index in range(layer_count)
            )
        )

    def unbind(self) -> tuple["GPTKVCache", ...]:
        """Split a batched cache into one cache per batch row."""
        if not self.layers:
            raise ValueError("cache must contain at least one layer")
        _ = self.seq_len
        batch_size = self.layers[0].key.size(0)
        if any(layer.key.size(0) != batch_size for layer in self.layers):
            raise ValueError("all layer caches must have the same batch size")
        return tuple(
            GPTKVCache(
                tuple(
                    AttentionKVCache(
                        key=layer.key[row_index : row_index + 1],
                        value=layer.value[row_index : row_index + 1],
                    )
                    for layer in self.layers
                )
            )
            for row_index in range(batch_size)
        )


class GPT(nn.Module):
    """Assemble the V1 decoder-only Transformer."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layers))
        self.final_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.apply(self._init_weights)
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Embedding, nn.Linear)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def _validate_input_ids(self, input_ids: torch.Tensor, cached_seq_len: int = 0) -> None:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, seq_len]")
        if input_ids.dtype != torch.long:
            raise ValueError("input_ids must have dtype torch.long")
        if input_ids.size(0) == 0 or input_ids.size(1) == 0:
            raise ValueError("batch and sequence dimensions must be non-empty")
        total_seq_len = cached_seq_len + input_ids.size(1)
        if total_seq_len > self.config.max_seq_len:
            raise ValueError(
                f"Sequence length {total_seq_len} exceeds max_seq_len={self.config.max_seq_len}"
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        self._validate_input_ids(input_ids)
        if targets is not None and targets.shape != input_ids.shape:
            raise ValueError("targets must have the same shape as input_ids")
        if targets is not None and targets.dtype != torch.long:
            raise ValueError("targets must have dtype torch.long")

        x = self.embedding_dropout(self.token_embedding(input_ids))
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.final_norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.config.vocab_size),
                targets.reshape(-1),
                ignore_index=IGNORE_INDEX,
            )
        return logits, loss

    def forward_with_cache(
        self,
        input_ids: torch.Tensor,
        cache: GPTKVCache | None = None,
    ) -> tuple[torch.Tensor, GPTKVCache]:
        """Return logits for new tokens and reusable per-layer K/V state."""
        if cache is not None and len(cache.layers) != self.config.n_layers:
            raise ValueError(
                f"cache must contain {self.config.n_layers} layers, got {len(cache.layers)}"
            )
        cached_seq_len = 0 if cache is None else cache.seq_len
        self._validate_input_ids(input_ids, cached_seq_len)

        x = self.embedding_dropout(self.token_embedding(input_ids))
        updated_layers: list[AttentionKVCache] = []
        for layer_index, block in enumerate(self.blocks):
            layer_cache = None if cache is None else cache.layers[layer_index]
            x, updated_cache = block.forward_with_cache(x, layer_cache)
            updated_layers.append(updated_cache)
        logits = self.lm_head(self.final_norm(x))
        return logits, GPTKVCache(tuple(updated_layers))
