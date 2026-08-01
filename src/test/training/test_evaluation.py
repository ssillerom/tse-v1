import math

import pytest
import tiktoken
import torch

from model.config import ModelConfig
from model.gpt import GPT
from training.evaluation import (
    EVALUATION_PROMPTS,
    EvaluationPrompt,
    evenly_spaced_validation_indices,
    generate_greedy,
    generate_prompt_samples,
    perplexity_from_loss,
)


def test_perplexity_converts_cross_entropy_back_to_effective_choices() -> None:
    assert perplexity_from_loss(0.0) == pytest.approx(1.0)
    assert perplexity_from_loss(math.log(8)) == pytest.approx(8.0)
    assert math.isinf(perplexity_from_loss(1_000.0))


def test_validation_indices_cover_the_complete_split_deterministically() -> None:
    assert evenly_spaced_validation_indices(dataset_size=10, sample_size=4) == (0, 3, 6, 9)
    assert evenly_spaced_validation_indices(dataset_size=3, sample_size=10) == (0, 1, 2)


class ScriptedGPT(GPT):
    """Return controlled next-token logits through the public GPT interface."""

    def __init__(
        self,
        vocab_size: int = 8,
        planned_tokens: tuple[int, ...] = (3, 4),
        padded_token_id: int | None = 7,
    ) -> None:
        super().__init__(
            ModelConfig(
                vocab_size=vocab_size,
                d_model=8,
                n_layers=1,
                n_heads=2,
                max_seq_len=8,
                tie_embeddings=False,
            )
        )
        self.planned_tokens = planned_tokens
        self.padded_token_id = padded_token_id
        self.call_index = 0

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        del targets
        logits = torch.zeros(
            input_ids.size(0),
            input_ids.size(1),
            self.config.vocab_size,
            device=input_ids.device,
        )
        planned_token = self.planned_tokens[min(self.call_index, len(self.planned_tokens) - 1)]
        if self.call_index == 0 and self.padded_token_id is not None:
            logits[:, -1, self.padded_token_id] = 100.0
        logits[:, -1, planned_token] = 10.0
        self.call_index += 1
        return logits, None


def test_greedy_generation_uses_real_tokenizer_vocabulary_and_stops_at_eot() -> None:
    model = ScriptedGPT()
    model.train()

    generated = generate_greedy(
        model=model,
        input_ids=torch.tensor([[1, 2]], dtype=torch.long),
        max_new_tokens=5,
        eot_token_id=4,
        tokenizer_vocab_size=6,
    )

    assert generated.tolist() == [[1, 2, 3, 4]]
    assert model.training


def test_greedy_generation_rejects_more_than_the_evaluation_token_limit() -> None:
    with pytest.raises(ValueError, match="at most 64"):
        generate_greedy(
            model=ScriptedGPT(),
            input_ids=torch.tensor([[1, 2]], dtype=torch.long),
            max_new_tokens=65,
            eot_token_id=4,
            tokenizer_vocab_size=6,
        )


def test_default_prompts_cover_fixed_english_continuation_categories() -> None:
    assert [(prompt.category, prompt.text) for prompt in EVALUATION_PROMPTS] == [
        ("narrative", "In a quiet town near the coast,"),
        ("science", "Photosynthesis is the process by which"),
        ("news", "The government announced this morning that"),
        ("argumentation", "One of the main advantages of solar energy is"),
        ("procedure", "To prepare a simple vegetable soup,"),
        ("dialogue", '"We should not be here," she said'),
    ]


def test_prompt_samples_decode_only_the_generated_continuation() -> None:
    encoding = tiktoken.get_encoding("gpt2")
    continuation_token = encoding.encode(" world")[0]
    model = ScriptedGPT(
        vocab_size=50_304,
        planned_tokens=(continuation_token, encoding.eot_token),
        padded_token_id=50_303,
    )

    samples = generate_prompt_samples(
        model=model,
        encoding=encoding,
        prompts=(EvaluationPrompt("test", "Hello"),),
        max_new_tokens=4,
        device="cpu",
    )

    assert len(samples) == 1
    assert samples[0].category == "test"
    assert samples[0].prompt == "Hello"
    assert samples[0].continuation == " world"
