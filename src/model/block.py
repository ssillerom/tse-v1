import torch
import torch.nn as nn

from src.model.attention import AttentionKVCache, MultiHeadAttention
from src.model.config import ModelConfig
from src.model.rms_norm import RMSNorm
from src.model.swiglu import SwiGLU


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        self.attention_norm = RMSNorm(
            d_model=config.d_model,
            eps=config.norm_eps,
        )

        self.attention = MultiHeadAttention(
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_kv_heads=config.n_kv_heads,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout,
            qkv_bias=config.qkv_bias,
            rope_theta=config.rope_theta,
            use_sdpa=config.use_sdpa,
        )

        self.mlp_norm = RMSNorm(
            d_model=config.d_model,
            eps=config.norm_eps,
        )

        self.mlp = SwiGLU(
            d_model=config.d_model,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x

    def forward_with_cache(
        self,
        x: torch.Tensor,
        cache: AttentionKVCache | None = None,
    ) -> tuple[torch.Tensor, AttentionKVCache]:
        attention_output, updated_cache = self.attention.forward_with_cache(
            self.attention_norm(x),
            cache,
        )
        x = x + attention_output
        x = x + self.mlp(self.mlp_norm(x))
        return x, updated_cache
