"""V1 decoder-only language model.

GPT assembles the model's components into one causal path:
token IDs -> embeddings -> transformer blocks -> final RMSNorm -> logits.
RoPE lives inside each attention layer, so this module does not add separate
positional embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.block import TransformerBlock
from src.model.config import ModelConfig
from src.model.rms_norm import RMSNorm


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

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, seq_len]")
        if input_ids.dtype != torch.long:
            raise ValueError("input_ids must have dtype torch.long")
        if input_ids.size(0) == 0 or input_ids.size(1) == 0:
            raise ValueError("batch and sequence dimensions must be non-empty")
        if input_ids.size(1) > self.config.max_seq_len:
            raise ValueError(
                f"Sequence length {input_ids.size(1)} exceeds max_seq_len={self.config.max_seq_len}"
            )
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
                ignore_index=-100,
            )
        return logits, loss
