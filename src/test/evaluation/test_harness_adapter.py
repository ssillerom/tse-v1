import math
from types import SimpleNamespace

import pytest
import tiktoken
import torch

from evaluation.harness_adapter import GPT2HarnessAdapter
from model.config import ModelConfig
from model.gpt import GPT
from test_support.cached_gpt import CachedGenerationGPT


class ConstantTokenGPT(CachedGenerationGPT):
    """Predict one real token while assigning a larger padded-vocab logit."""

    def __init__(self, predicted_token: int, max_seq_len: int = 8) -> None:
        super().__init__(
            ModelConfig(
                vocab_size=50_304,
                d_model=8,
                n_layers=1,
                n_heads=2,
                max_seq_len=max_seq_len,
                tie_embeddings=False,
            )
        )
        self.predicted_token = predicted_token
        self.observed_batch_sizes: list[int] = []

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        del targets
        self.observed_batch_sizes.append(input_ids.size(0))
        logits = torch.zeros(
            input_ids.size(0),
            input_ids.size(1),
            self.config.vocab_size,
            device=input_ids.device,
        )
        logits[..., 50_303] = 100.0
        logits[..., self.predicted_token] = 30.0
        return logits, None


class UniformGPT(GPT):
    def __init__(self, max_seq_len: int = 4) -> None:
        super().__init__(
            ModelConfig(
                vocab_size=50_304,
                d_model=8,
                n_layers=1,
                n_heads=2,
                max_seq_len=max_seq_len,
                tie_embeddings=False,
            )
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        del targets
        return (
            torch.zeros(
                input_ids.size(0),
                input_ids.size(1),
                self.config.vocab_size,
                device=input_ids.device,
            ),
            None,
        )


class PlannedGenerationGPT(ConstantTokenGPT):
    def __init__(self, planned_tokens: tuple[int, ...]) -> None:
        super().__init__(predicted_token=planned_tokens[0])
        self.planned_tokens = planned_tokens
        self.generation_step = 0

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        self.predicted_token = self.planned_tokens[
            min(self.generation_step, len(self.planned_tokens) - 1)
        ]
        self.generation_step += 1
        return super().forward(input_ids, targets)


def test_adapter_tokenizes_context_and_continuation_as_one_bpe_sequence() -> None:
    encoding = tiktoken.get_encoding("gpt2")
    adapter = GPT2HarnessAdapter(
        model=UniformGPT(),
        encoding=encoding,
        device="cpu",
    )

    context_ids, continuation_ids = adapter.encode_pair("answer", "ing")

    assert context_ids == [41_484]
    assert continuation_ids == [86, 1_586]


def test_loglikelihood_batches_requests_and_masks_padded_model_vocabulary() -> None:
    encoding = tiktoken.get_encoding("gpt2")
    quote_token = encoding.encode('"')
    assert quote_token == [1]
    model = ConstantTokenGPT(predicted_token=quote_token[0])
    adapter = GPT2HarnessAdapter(
        model=model,
        encoding=encoding,
        device="cpu",
        batch_size=2,
    )
    requests = [
        SimpleNamespace(args=("a", '"')),
        SimpleNamespace(args=("b", '"')),
    ]

    results = adapter.loglikelihood(requests)

    assert model.observed_batch_sizes == [2]
    assert all(score > -1e-6 for score, _ in results)
    assert [is_greedy for _, is_greedy in results] == [True, True]


def test_rolling_loglikelihood_scores_every_token_exactly_once() -> None:
    encoding = tiktoken.get_encoding("gpt2")
    text = "one two three four five six seven"
    assert len(encoding.encode(text)) == 7
    adapter = GPT2HarnessAdapter(
        model=UniformGPT(max_seq_len=4),
        encoding=encoding,
        device="cpu",
        batch_size=2,
    )

    results = adapter.loglikelihood_rolling([SimpleNamespace(args=(text,))])

    assert results == pytest.approx([-7 * math.log(encoding.n_vocab)])


def test_generate_until_returns_text_before_the_first_stop_sequence() -> None:
    encoding = tiktoken.get_encoding("gpt2")
    hello_token = encoding.encode(" hello")
    newline_token = encoding.encode("\n")
    assert len(hello_token) == 1
    assert len(newline_token) == 1
    adapter = GPT2HarnessAdapter(
        model=PlannedGenerationGPT((hello_token[0], newline_token[0])),
        encoding=encoding,
        device="cpu",
    )

    results = adapter.generate_until(
        [
            SimpleNamespace(
                args=("a", {"until": ["\n"], "max_gen_toks": 4}),
            )
        ]
    )

    assert results == [" hello"]


def test_generate_until_never_exceeds_the_request_token_budget() -> None:
    encoding = tiktoken.get_encoding("gpt2")
    hello_token = encoding.encode(" hello")
    world_token = encoding.encode(" world")
    assert len(hello_token) == 1
    assert len(world_token) == 1
    model = PlannedGenerationGPT((hello_token[0], world_token[0]))
    adapter = GPT2HarnessAdapter(
        model=model,
        encoding=encoding,
        device="cpu",
    )

    results = adapter.generate_until(
        [SimpleNamespace(args=("a", {"until": [], "max_gen_toks": 1}))]
    )

    assert results == [" hello"]
    assert model.generation_step == 1


def test_generate_until_batches_requests_with_matching_cache_lengths() -> None:
    encoding = tiktoken.get_encoding("gpt2")
    hello_token = encoding.encode(" hello")
    newline_token = encoding.encode("\n")
    model = PlannedGenerationGPT((hello_token[0], newline_token[0]))
    adapter = GPT2HarnessAdapter(
        model=model,
        encoding=encoding,
        device="cpu",
        batch_size=2,
    )
    requests = [
        SimpleNamespace(args=(context, {"until": ["\n"], "max_gen_toks": 4}))
        for context in ("a", "b")
    ]

    results = adapter.generate_until(requests)

    assert results == [" hello", " hello"]
    assert model.observed_batch_sizes == [2, 2]
