import math
from dataclasses import dataclass
from numbers import Real


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for the V1 causal language model."""

    vocab_size: int = 50_304
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 8
    max_seq_len: int = 256
    dropout: float = 0.0
    qkv_bias: bool = False
    rope_theta: float = 10_000.0
    norm_eps: float = 1e-6
    tie_embeddings: bool = True

    def __post_init__(self) -> None:
        integer_settings = {
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "max_seq_len": self.max_seq_len,
        }
        for name, value in integer_settings.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")

        head_dim = self.head_dim()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")

        if (
            not isinstance(self.dropout, Real)
            or isinstance(self.dropout, bool)
            or not math.isfinite(float(self.dropout))
            or not 0.0 <= self.dropout < 1.0
        ):
            raise ValueError(f"dropout must be in [0.0, 1.0), got {self.dropout}")

        real_settings = {
            "rope_theta": self.rope_theta,
            "norm_eps": self.norm_eps,
        }
        for name, value in real_settings.items():
            if (
                not isinstance(value, Real)
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or value <= 0.0
            ):
                raise ValueError(f"{name} must be finite and positive, got {value!r}")

        if not isinstance(self.qkv_bias, bool):
            raise ValueError(f"qkv_bias must be a boolean, got {self.qkv_bias!r}")
        if not isinstance(self.tie_embeddings, bool):
            raise ValueError(f"tie_embeddings must be a boolean, got {self.tie_embeddings!r}")

    def head_dim(self) -> int:
        """Return the dimension of each attention head."""
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model {self.d_model} is not divisible by n_heads {self.n_heads}")
        return self.d_model // self.n_heads
