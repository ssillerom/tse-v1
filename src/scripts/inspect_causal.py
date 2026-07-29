import tiktoken
import torch
from torch import nn

from src.model.attention import MultiHeadAttention
from src.model.config import ModelConfig

config = ModelConfig()

tokenizer = tiktoken.get_encoding("gpt2")

token_ids = tokenizer.encode("Your journey starts with one step")

input_ids = torch.tensor(
    [token_ids],
    dtype=torch.long,
)

print("Input token IDs:", input_ids)

token_embeddings = nn.Embedding(
    num_embeddings=config.vocab_size,
    embedding_dim=config.d_model,
)

x = token_embeddings(input_ids)

print("Input embeddings:", x.shape)
print("Input embeddings:", x)

attention_layer = MultiHeadAttention(
    d_model=config.d_model,
    n_heads=config.n_heads,
    max_seq_len=config.max_seq_len,
    dropout=config.dropout,
    qkv_bias=config.qkv_bias,
    rope_theta=config.rope_theta,
)

attention_layer.eval()

context_vectors = attention_layer(x)

print("Token IDs:", input_ids.shape)
print("Embeddings:", x.shape)
print("Context vectors:", context_vectors.shape)
print("context_vectors:", context_vectors)
