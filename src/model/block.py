import torch
import torch.nn as nn

from src.model.attention import MultiHeadAttention
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
            max_seq_len=config.max_seq_len,
            dropout=config.dropout,
            qkv_bias=config.qkv_bias,
            rope_theta=config.rope_theta,
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
