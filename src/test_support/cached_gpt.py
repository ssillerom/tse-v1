"""Reusable GPT test double support for cached generation."""

import torch

from src.model.attention import AttentionKVCache
from src.model.gpt import GPT, GPTKVCache


class CachedGenerationGPT(GPT):
    """Adapt a scripted ``forward`` implementation to the public cache interface."""

    def forward_with_cache(
        self,
        input_ids: torch.Tensor,
        cache: GPTKVCache | None = None,
    ) -> tuple[torch.Tensor, GPTKVCache]:
        logits, _ = self.forward(input_ids)
        cached_tokens = (0 if cache is None else cache.seq_len) + input_ids.size(1)
        n_kv_heads = self.config.n_kv_heads or self.config.n_heads
        key = torch.zeros(
            input_ids.size(0),
            n_kv_heads,
            cached_tokens,
            self.config.head_dim(),
            device=input_ids.device,
        )
        layers = tuple(
            AttentionKVCache(key=key.clone(), value=key.clone())
            for _ in range(self.config.n_layers)
        )
        return logits, GPTKVCache(layers)
