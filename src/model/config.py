from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for a causal language model."""

    vocab_size: int = 50_304
    d_model: int = 256
    n_heads: int = 8
    max_seq_len: int = 256
    dropout: float = 0.1
    attention_bias: bool = False
    rope_theta: float = 10_000.0

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {self.vocab_size}")
        if self.d_model <= 0:
            raise ValueError(f"d_model must be positive, got {self.d_model}")
        if self.n_heads <= 0:
            raise ValueError(f"n_heads must be positive, got {self.n_heads}")
        if self.max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {self.max_seq_len}")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(f"dropout must be in [0.0, 1.0), got {self.dropout}")
        if self.rope_theta <= 0.0:
            raise ValueError(f"rope_theta must be positive, got {self.rope_theta}")

    def head_dim(self) -> int:
        """Return the dimension of each attention head."""
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model {self.d_model} is not divisible by n_heads {self.n_heads}")
        return self.d_model // self.n_heads
