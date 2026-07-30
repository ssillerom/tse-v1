import json
from pathlib import Path
from typing import Any

import torch

from model.config import ModelConfig
from model.gpt import GPT
from scripts.eval_checkpoint import HarnessRuntime, main
from training.checkpoint import save_checkpoint
from training.trainer import TrainingConfig


def test_eval_checkpoint_cli_writes_a_reproducible_atomic_artifact(tmp_path: Path) -> None:
    model = GPT(
        ModelConfig(
            vocab_size=50_304,
            d_model=8,
            n_layers=1,
            n_heads=2,
            max_seq_len=8,
        )
    )
    checkpoint_path = save_checkpoint(
        path=tmp_path / "step_000003.pt",
        model=model,
        optimizer=torch.optim.AdamW(model.parameters()),
        step=3,
        training_config=TrainingConfig(max_steps=4),
    )
    output_path = tmp_path / "results" / "evaluation.json"
    received: dict[str, Any] = {}

    def simple_evaluate(**kwargs: object) -> dict[str, object]:
        received.update(kwargs)
        return {"results": {"hellaswag": {"acc,none": 0.25}}}

    runtime = HarnessRuntime(
        simple_evaluate=simple_evaluate,
        handle_non_serializable=str,
        version="0.4.test",
    )

    exit_code = main(
        [
            "--checkpoint",
            str(checkpoint_path),
            "--output",
            str(output_path),
            "--tasks",
            "hellaswag,arc_easy",
            "--num-fewshot",
            "2",
            "--batch-size",
            "4",
            "--limit",
            "3",
            "--device",
            "cpu",
        ],
        harness_runtime=runtime,
    )

    artifact = json.loads(output_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert received["tasks"] == ["hellaswag", "arc_easy"]
    assert received["num_fewshot"] == 2
    assert received["batch_size"] == 4
    assert received["limit"] == 3
    assert artifact["checkpoint"]["step"] == 3
    assert len(artifact["checkpoint"]["sha256"]) == 64
    assert artifact["model_config"]["d_model"] == 8
    assert artifact["evaluation"]["tasks"] == ["hellaswag", "arc_easy"]
    assert artifact["versions"]["lm_eval"] == "0.4.test"
    assert artifact["repository"].keys() == {"commit", "dirty"}
    assert artifact["harness"]["results"]["hellaswag"]["acc,none"] == 0.25
    assert not output_path.with_name(f"{output_path.name}.tmp").exists()
