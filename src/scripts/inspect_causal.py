"""Show the smallest token IDs -> embeddings -> causal-attention forward path."""

import tiktoken
import torch
from torch import nn

from src.model.attention import MultiHeadAttention
from src.model.config import ModelConfig


def main() -> int:
    """Run one inspectable attention example without training any parameters."""
    config = ModelConfig()
    tokenizer = tiktoken.get_encoding("gpt2")

    # Phase 1: turn human text into the integer vocabulary consumed by GPT.
    token_ids = tokenizer.encode("Your journey starts with one step")
    input_ids = torch.tensor([token_ids], dtype=torch.long)
    print("Input token IDs:", input_ids)

    # Phase 2: look up one learned-width vector for each token position.
    token_embeddings = nn.Embedding(
        num_embeddings=config.vocab_size,
        embedding_dim=config.d_model,
    )
    embeddings = token_embeddings(input_ids)
    print("Input embeddings:", embeddings.shape)
    print("Input embeddings:", embeddings)

    # Phase 3: causal attention mixes each vector with its visible prefix.
    attention_layer = MultiHeadAttention(
        d_model=config.d_model,
        n_heads=config.n_heads,
        max_seq_len=config.max_seq_len,
        dropout=config.dropout,
        qkv_bias=config.qkv_bias,
        rope_theta=config.rope_theta,
    )
    attention_layer.eval()
    with torch.no_grad():
        context_vectors = attention_layer(embeddings)

    print("Token IDs:", input_ids.shape)
    print("Embeddings:", embeddings.shape)
    print("Context vectors:", context_vectors.shape)
    print("Context vectors:", context_vectors)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
