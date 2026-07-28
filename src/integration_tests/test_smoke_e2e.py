import math
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from data import prepare_data
from data.dataset import PretrainingDataset
from data.manifest import load_manifest
from model.config import ModelConfig
from model.gpt import GPT
from training.checkpoint import load_checkpoint, save_checkpoint
from training.trainer import TrainingConfig, evaluate, train


class TinyEncoding:
    """Deterministic local tokenizer used only at the external tokenizer seam."""

    eot_token = 0
    max_token_value = 7
    token_ids = {"a": 1, "b": 2, "c": 3}

    def encode(
        self,
        text: str,
        disallowed_special: tuple[()] = (),
    ) -> list[int]:
        del disallowed_special
        return [self.token_ids[token] for token in text.split()]


def test_smoke_e2e(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Prove the local data-to-resumable-training path works end to end."""
    torch.manual_seed(42)
    train_documents = [
        {"text": "a b a b a b a b"},
        {"text": "c b c b c b c b"},
    ] * 12
    validation_documents = [
        {"text": "a b a b a b a b"},
        {"text": "c b c b c b c b"},
    ] * 4

    def local_document_stream(**arguments: object) -> list[dict[str, str]]:
        return train_documents if arguments["split"] == "train" else validation_documents

    def local_encoding(encoding_name: str) -> TinyEncoding:
        if encoding_name != "tiny":
            raise ValueError(f"unexpected encoding: {encoding_name}")
        return TinyEncoding()

    monkeypatch.setattr(
        prepare_data,
        "load_streaming_hf_dataset",
        local_document_stream,
    )
    monkeypatch.setattr(prepare_data.tiktoken, "get_encoding", local_encoding)

    train_data_dir = tmp_path / "prepared_train"
    validation_data_dir = tmp_path / "prepared_validation"
    prepare_data.prepare_streaming_dataset(
        output_dir=train_data_dir,
        dataset_name="local/smoke",
        text_field="text",
        num_tokens=256,
        shard_size=64,
        split="train",
        encoding_name="tiny",
    )
    prepare_data.prepare_streaming_dataset(
        output_dir=validation_data_dir,
        dataset_name="local/smoke",
        text_field="text",
        num_tokens=128,
        shard_size=64,
        split="validation",
        encoding_name="tiny",
    )

    train_manifest = train_data_dir / "manifest.json"
    validation_manifest = validation_data_dir / "manifest.json"
    assert train_manifest.is_file()
    assert validation_manifest.is_file()
    prepared_manifest = load_manifest(train_manifest)
    first_train_shard = prepared_manifest.shards_for_split("train")[0]
    first_document_tokens = np.fromfile(
        first_train_shard.path,
        dtype=np.uint16,
        count=9,
    )
    expected_first_document = TinyEncoding().encode(train_documents[0]["text"]) + [
        TinyEncoding.eot_token
    ]
    assert first_document_tokens.tolist() == expected_first_document

    sequence_length = 6
    train_dataset = PretrainingDataset(
        manifest_path=train_manifest,
        split="train",
        seq_len=sequence_length,
    )
    validation_dataset = PretrainingDataset(
        manifest_path=validation_manifest,
        split="validation",
        seq_len=sequence_length,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=False,
        drop_last=True,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=2,
        shuffle=False,
        drop_last=True,
    )
    first_inputs, first_targets = next(iter(train_loader))
    assert first_inputs.shape == (2, sequence_length)
    assert first_targets.shape == (2, sequence_length)
    assert first_inputs.dtype == torch.long
    assert first_targets.dtype == torch.long
    torch.testing.assert_close(first_inputs[:, 1:], first_targets[:, :-1])

    model_config = ModelConfig(
        vocab_size=8,
        d_model=16,
        n_layers=2,
        n_heads=4,
        max_seq_len=sequence_length,
        dropout=0.0,
        use_sdpa=True,
    )
    model = GPT(model_config)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2e-2,
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )
    checkpoint_step = 30
    training_config = TrainingConfig(
        max_steps=31,
        grad_accum_steps=2,
        warmup_steps=2,
        max_learning_rate=2e-2,
        min_learning_rate=2e-3,
        max_grad_norm=1.0,
        eval_interval=10,
        eval_batches=3,
    )
    embedding_before = model.token_embedding.weight.detach().clone()
    initial_validation_loss = evaluate(
        model=model,
        batches=validation_loader,
        device="cpu",
        max_batches=training_config.eval_batches,
    )

    history = train(
        model=model,
        optimizer=optimizer,
        train_batches=train_loader,
        config=training_config,
        device="cpu",
        validation_batches=validation_loader,
        end_step=checkpoint_step,
    )
    final_validation_loss = evaluate(
        model=model,
        batches=validation_loader,
        device="cpu",
        max_batches=training_config.eval_batches,
    )

    assert math.isfinite(initial_validation_loss)
    assert abs(initial_validation_loss - math.log(model_config.vocab_size)) < 0.5
    assert len(history) == checkpoint_step
    assert all(math.isfinite(metrics.loss) for metrics in history)
    assert all(
        math.isfinite(metrics.gradient_norm) and metrics.gradient_norm > 0.0 for metrics in history
    )
    assert history[0].learning_rate == pytest.approx(training_config.min_learning_rate)
    assert history[training_config.warmup_steps].learning_rate == pytest.approx(
        training_config.max_learning_rate
    )
    assert history[-1].learning_rate < training_config.max_learning_rate
    assert [metrics.step for metrics in history if metrics.validation_loss is not None] == [
        10,
        20,
        30,
    ]
    assert not torch.equal(model.token_embedding.weight, embedding_before)
    assert final_validation_loss < initial_validation_loss

    checkpoint_path = save_checkpoint(
        path=tmp_path / "checkpoints" / "step_000030.pt",
        model=model,
        optimizer=optimizer,
        step=checkpoint_step,
        training_config=training_config,
    )
    restored_model = GPT(model_config)
    restored_optimizer = torch.optim.AdamW(
        restored_model.parameters(),
        lr=training_config.max_learning_rate,
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )
    loaded_step = load_checkpoint(
        path=checkpoint_path,
        model=restored_model,
        optimizer=restored_optimizer,
        training_config=training_config,
        map_location="cpu",
    )

    model.eval()
    restored_model.eval()
    expected_logits, _ = model(first_inputs)
    restored_logits, _ = restored_model(first_inputs)
    assert loaded_step == checkpoint_step
    torch.testing.assert_close(restored_logits, expected_logits, rtol=0.0, atol=0.0)

    original_resume_history = train(
        model=model,
        optimizer=optimizer,
        train_batches=train_loader,
        config=training_config,
        device="cpu",
        validation_batches=validation_loader,
        start_step=loaded_step,
    )
    restored_resume_history = train(
        model=restored_model,
        optimizer=restored_optimizer,
        train_batches=train_loader,
        config=training_config,
        device="cpu",
        validation_batches=validation_loader,
        start_step=loaded_step,
    )

    assert [metrics.step for metrics in original_resume_history] == [31]
    assert [metrics.step for metrics in restored_resume_history] == [31]
    for expected, restored in zip(
        model.state_dict().values(),
        restored_model.state_dict().values(),
        strict=True,
    ):
        torch.testing.assert_close(restored, expected, rtol=0.0, atol=0.0)
