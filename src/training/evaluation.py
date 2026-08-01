"""Quantitative and qualitative evaluation helpers for V1 training."""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real

import tiktoken
import torch

from src.model.gpt import GPT

MAX_EVALUATION_NEW_TOKENS = 64


@dataclass(frozen=True)
class EvaluationPrompt:
    """One stable prompt used to compare generations across checkpoints."""

    category: str
    text: str


@dataclass(frozen=True)
class GenerationSample:
    """A decoded continuation generated for one fixed evaluation prompt."""

    category: str
    prompt: str
    continuation: str


EVALUATION_PROMPTS = (
    EvaluationPrompt("narrative", "In a quiet town near the coast,"),
    EvaluationPrompt("science", "Photosynthesis is the process by which"),
    EvaluationPrompt("news", "The government announced this morning that"),
    EvaluationPrompt(
        "argumentation",
        "One of the main advantages of solar energy is",
    ),
    EvaluationPrompt("procedure", "To prepare a simple vegetable soup,"),
    EvaluationPrompt("dialogue", '"We should not be here," she said'),
)


def evenly_spaced_validation_indices(dataset_size: int, sample_size: int) -> tuple[int, ...]:
    """Choose a stable validation subset that spans the complete dataset."""
    if not isinstance(dataset_size, int) or isinstance(dataset_size, bool) or dataset_size <= 0:
        raise ValueError("dataset_size must be a positive integer")
    if not isinstance(sample_size, int) or isinstance(sample_size, bool) or sample_size <= 0:
        raise ValueError("sample_size must be a positive integer")
    if sample_size >= dataset_size:
        return tuple(range(dataset_size))
    if sample_size == 1:
        return (dataset_size // 2,)

    # Integer arithmetic keeps the selection identical across Python and
    # NumPy versions while including both ends of the validation split.
    last_index = dataset_size - 1
    return tuple(
        sample_index * last_index // (sample_size - 1) for sample_index in range(sample_size)
    )


def perplexity_from_loss(loss: float) -> float:
    """Convert mean cross-entropy in nats to perplexity."""
    if (
        not isinstance(loss, Real)
        or isinstance(loss, bool)
        or not math.isfinite(float(loss))
        or loss < 0.0
    ):
        raise ValueError(f"loss must be finite and non-negative, got {loss!r}")
    try:
        return math.exp(loss)
    except OverflowError:
        return math.inf


def generate_greedy(
    model: GPT,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    eot_token_id: int,
    tokenizer_vocab_size: int,
) -> torch.Tensor:
    """Generate one deterministic continuation and stop after EOT."""
    if input_ids.ndim != 2 or input_ids.size(0) != 1 or input_ids.size(1) == 0:
        raise ValueError("input_ids must have shape [1, prompt_length] with a non-empty prompt")
    if input_ids.dtype != torch.long:
        raise ValueError("input_ids must have dtype torch.long")
    if (
        not isinstance(max_new_tokens, int)
        or isinstance(max_new_tokens, bool)
        or max_new_tokens <= 0
        or max_new_tokens > MAX_EVALUATION_NEW_TOKENS
    ):
        raise ValueError(
            f"max_new_tokens must be a positive integer of at most "
            f"{MAX_EVALUATION_NEW_TOKENS}, got {max_new_tokens!r}"
        )
    if (
        not isinstance(tokenizer_vocab_size, int)
        or isinstance(tokenizer_vocab_size, bool)
        or not 0 < tokenizer_vocab_size <= model.config.vocab_size
    ):
        raise ValueError(
            "tokenizer_vocab_size must be between 1 and the model vocabulary size, "
            f"got {tokenizer_vocab_size!r}"
        )
    if (
        not isinstance(eot_token_id, int)
        or isinstance(eot_token_id, bool)
        or not 0 <= eot_token_id < tokenizer_vocab_size
    ):
        raise ValueError(
            f"eot_token_id must belong to the tokenizer vocabulary, got {eot_token_id!r}"
        )

    was_training = model.training
    model.eval()
    generated = input_ids
    try:
        with torch.inference_mode():
            for _ in range(max_new_tokens):
                context = generated[:, -model.config.max_seq_len :]
                logits, _ = model(context)
                next_token_logits = logits[:, -1, :tokenizer_vocab_size]
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                generated = torch.cat((generated, next_token), dim=1)
                if next_token.item() == eot_token_id:
                    break
    finally:
        model.train(was_training)
    return generated


def generate_prompt_samples(
    model: GPT,
    encoding: tiktoken.Encoding,
    prompts: Sequence[EvaluationPrompt] = EVALUATION_PROMPTS,
    max_new_tokens: int = 64,
    device: torch.device | str = "cpu",
) -> tuple[GenerationSample, ...]:
    """Generate deterministic continuations for a stable prompt suite."""
    samples: list[GenerationSample] = []
    for prompt in prompts:
        prompt_token_ids = encoding.encode(prompt.text, disallowed_special=())
        if not prompt_token_ids:
            raise ValueError(
                f"evaluation prompt must encode to at least one token: {prompt.text!r}"
            )
        input_ids = torch.tensor(
            [prompt_token_ids],
            dtype=torch.long,
            device=device,
        )
        output_ids = generate_greedy(
            model=model,
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            eot_token_id=encoding.eot_token,
            tokenizer_vocab_size=encoding.n_vocab,
        )
        continuation_ids = output_ids[0, len(prompt_token_ids) :].tolist()
        if continuation_ids and continuation_ids[-1] == encoding.eot_token:
            continuation_ids.pop()
        samples.append(
            GenerationSample(
                category=prompt.category,
                prompt=prompt.text,
                continuation=encoding.decode(continuation_ids),
            )
        )
    return tuple(samples)
