import math
from collections.abc import Mapping

import pytest
import tiktoken
import torch
import wandb

from model.config import ModelConfig
from test_support.cached_gpt import CachedGenerationGPT
from training.evaluation import EvaluationPrompt
from training.trainer import EvaluationMetrics, StepMetrics
from training.wandb_logging import WandbEvaluationLogger, metrics_to_wandb


def test_wandb_metrics_include_validation_only_on_evaluation_steps() -> None:
    training_metrics = StepMetrics(
        step=1,
        loss=10.5,
        gradient_norm=0.8,
        learning_rate=3e-5,
        tokens_in_step=8_192,
        tokens_seen=8_192,
        step_time_seconds=0.5,
        tokens_per_second=16_384.0,
    )
    evaluation_metrics = StepMetrics(
        step=50,
        loss=7.0,
        gradient_norm=0.3,
        learning_rate=3e-4,
        tokens_in_step=8_192,
        tokens_seen=409_600,
        step_time_seconds=0.4,
        tokens_per_second=20_480.0,
        validation_loss=7.2,
        validation_perplexity=1339.430764,
    )

    assert metrics_to_wandb(training_metrics) == {
        "trainer/global_step": 1,
        "train/loss": 10.5,
        "optimizer/gradient_norm": 0.8,
        "optimizer/learning_rate": 3e-5,
        "trainer/tokens_in_step": 8_192,
        "trainer/tokens_seen": 8_192,
        "performance/step_time_seconds": 0.5,
        "performance/tokens_per_second": 16_384.0,
    }
    assert metrics_to_wandb(evaluation_metrics) == {
        "trainer/global_step": 50,
        "train/loss": 7.0,
        "optimizer/gradient_norm": 0.3,
        "optimizer/learning_rate": 3e-4,
        "trainer/tokens_in_step": 8_192,
        "trainer/tokens_seen": 409_600,
        "performance/step_time_seconds": 0.4,
        "performance/tokens_per_second": 20_480.0,
        "validation/loss": 7.2,
        "validation/perplexity": 1339.430764,
    }


def test_wandb_metrics_include_each_validation_domain() -> None:
    metrics = StepMetrics(
        step=50,
        loss=7.0,
        gradient_norm=0.3,
        learning_rate=3e-4,
        tokens_in_step=8_192,
        tokens_seen=409_600,
        source_tokens_seen={"web": 327_680, "math": 81_920},
        step_time_seconds=0.4,
        tokens_per_second=20_480.0,
        validation_loss=2.0,
        validation_perplexity=math.exp(2.0),
        validation_domains={
            "web": EvaluationMetrics(loss=1.0, perplexity=math.e, target_tokens=1_024),
            "math": EvaluationMetrics(
                loss=5.0,
                perplexity=math.exp(5.0),
                target_tokens=512,
            ),
        },
    )

    payload = metrics_to_wandb(metrics)

    assert payload["validation/web/loss"] == 1.0
    assert payload["validation/web/perplexity"] == pytest.approx(math.e)
    assert payload["validation/web/target_tokens"] == 1_024
    assert payload["validation/math/loss"] == 5.0
    assert payload["validation/math/target_tokens"] == 512
    assert payload["trainer/source_tokens_seen/web"] == 327_680
    assert payload["trainer/source_tokens_seen/math"] == 81_920


class FakeRun:
    def __init__(self) -> None:
        self.logged: list[dict[str, object]] = []
        self.defined: list[tuple[str, dict[str, object]]] = []

    def log(self, data: Mapping[str, object]) -> None:
        self.logged.append(dict(data))

    def define_metric(self, name: str, **kwargs: object) -> object:
        self.defined.append((name, kwargs))
        return object()


class ScriptedGPT(CachedGenerationGPT):
    def __init__(self, continuation_token: int, eot_token: int) -> None:
        super().__init__(
            ModelConfig(
                vocab_size=50_304,
                d_model=8,
                n_layers=1,
                n_heads=2,
                max_seq_len=8,
                tie_embeddings=False,
            )
        )
        self.tokens = (continuation_token, eot_token)
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
        next_token = self.tokens[self.call_index % len(self.tokens)]
        logits[:, -1, next_token] = 10.0
        self.call_index += 1
        return logits, None


def test_wandb_logger_metric_definitions_are_accepted_by_the_sdk(tmp_path) -> None:
    encoding = tiktoken.get_encoding("gpt2")
    model = ScriptedGPT(
        continuation_token=encoding.encode(" world")[0],
        eot_token=encoding.eot_token,
    )

    with wandb.init(
        project="wandb-logger-regression",
        mode="offline",
        dir=tmp_path,
        settings=wandb.Settings(silent=True),
    ) as run:
        WandbEvaluationLogger(
            run=run,
            model=model,
            encoding=encoding,
            device="cpu",
            validation_domains=("web", "math"),
        )


def test_wandb_logger_generates_a_fixed_prompt_table_only_at_its_interval() -> None:
    encoding = tiktoken.get_encoding("gpt2")
    model = ScriptedGPT(
        continuation_token=encoding.encode(" world")[0],
        eot_token=encoding.eot_token,
    )
    run = FakeRun()
    logger = WandbEvaluationLogger(
        run=run,
        model=model,
        encoding=encoding,
        device="cpu",
        sample_interval=2,
        max_new_tokens=4,
        prompts=(EvaluationPrompt("test", "Hello"),),
        validation_domains=("web", "math"),
    )

    assert (
        "validation/loss",
        {"step_metric": "trainer/global_step", "summary": "min"},
    ) in run.defined
    assert (
        "validation/web/loss",
        {"step_metric": "trainer/global_step", "summary": "min"},
    ) in run.defined
    assert (
        "validation/math/perplexity",
        {"step_metric": "trainer/global_step", "summary": "min"},
    ) in run.defined
    assert (
        "validation/math/target_tokens",
        {"step_metric": "trainer/global_step"},
    ) in run.defined
    logger.log_samples(step=0)
    logger(
        StepMetrics(
            step=1,
            loss=10.0,
            gradient_norm=0.5,
            learning_rate=3e-5,
            tokens_in_step=32,
            tokens_seen=32,
            step_time_seconds=0.1,
            tokens_per_second=320.0,
        )
    )
    logger(
        StepMetrics(
            step=2,
            loss=9.0,
            gradient_norm=0.4,
            learning_rate=6e-5,
            tokens_in_step=32,
            tokens_seen=64,
            step_time_seconds=0.1,
            tokens_per_second=320.0,
            validation_loss=9.1,
            validation_perplexity=8955.292703,
        )
    )

    assert len(run.logged) == 4
    initial_table = run.logged[0]["samples/fixed_prompts"]
    assert initial_table.data == [[0, "test", "Hello", " world"]]
    assert "samples/fixed_prompts" not in run.logged[1]
    table = run.logged[3]["samples/fixed_prompts"]
    assert table.data == [[2, "test", "Hello", " world"]]
