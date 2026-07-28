import math

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from model.config import ModelConfig
from model.gpt import GPT
from training.trainer import TrainingConfig, evaluate, train, train_step


def _tiny_model() -> GPT:
    return GPT(
        ModelConfig(
            vocab_size=8,
            d_model=16,
            n_layers=2,
            n_heads=4,
            max_seq_len=6,
            dropout=0.0,
        )
    )


def test_train_step_produces_finite_metrics_and_updates_parameters() -> None:
    torch.manual_seed(42)
    model = _tiny_model()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-2,
        weight_decay=0.0,
    )
    batch = (
        torch.tensor([[0, 1, 2, 3, 4, 5]]),
        torch.tensor([[1, 2, 3, 4, 5, 6]]),
    )
    embedding_before = model.token_embedding.weight.detach().clone()

    loss, gradient_norm = train_step(
        model=model,
        optimizer=optimizer,
        microbatches=[batch],
        max_grad_norm=1.0,
        device="cpu",
    )

    assert math.isfinite(loss)
    assert math.isfinite(gradient_norm)
    assert gradient_norm > 0.0
    assert not torch.equal(model.token_embedding.weight, embedding_before)


def test_train_step_accumulation_matches_one_combined_batch() -> None:
    torch.manual_seed(42)
    accumulated_model = _tiny_model()
    combined_model = _tiny_model()
    combined_model.load_state_dict(accumulated_model.state_dict())
    accumulated_optimizer = torch.optim.SGD(accumulated_model.parameters(), lr=1e-2)
    combined_optimizer = torch.optim.SGD(combined_model.parameters(), lr=1e-2)
    first_batch = (
        torch.tensor([[0, 1, 2, 3, 4, 5]]),
        torch.tensor([[1, 2, 3, 4, 5, 6]]),
    )
    second_batch = (
        torch.tensor(
            [
                [1, 2, 3, 4, 5, 6],
                [2, 3, 4, 5, 6, 7],
            ]
        ),
        torch.tensor(
            [
                [2, 3, 4, 5, 6, 7],
                [3, 4, 5, 6, 7, 0],
            ]
        ),
    )
    combined_batch = (
        torch.cat((first_batch[0], second_batch[0])),
        torch.cat((first_batch[1], second_batch[1])),
    )

    train_step(
        model=accumulated_model,
        optimizer=accumulated_optimizer,
        microbatches=[first_batch, second_batch],
        max_grad_norm=1e6,
        device="cpu",
    )
    train_step(
        model=combined_model,
        optimizer=combined_optimizer,
        microbatches=[combined_batch],
        max_grad_norm=1e6,
        device="cpu",
    )

    for accumulated_parameter, combined_parameter in zip(
        accumulated_model.parameters(),
        combined_model.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(accumulated_parameter, combined_parameter)


def test_train_step_rejects_boolean_gradient_norm_limit() -> None:
    model = _tiny_model()
    optimizer = torch.optim.AdamW(model.parameters())
    batch = (
        torch.tensor([[0, 1, 2, 3, 4, 5]]),
        torch.tensor([[1, 2, 3, 4, 5, 6]]),
    )

    with pytest.raises(ValueError, match="max_grad_norm must be finite and positive"):
        train_step(
            model=model,
            optimizer=optimizer,
            microbatches=[batch],
            max_grad_norm=True,
            device="cpu",
        )


def test_evaluate_returns_loss_without_gradients_and_restores_training_mode() -> None:
    torch.manual_seed(42)
    model = _tiny_model()
    model.train()
    input_ids = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5, 6],
        ]
    )
    targets = torch.tensor(
        [
            [1, 2, 3, 4, 5, 6],
            [2, 3, 4, 5, 6, 7],
        ]
    )
    loader = DataLoader(TensorDataset(input_ids, targets), batch_size=1)

    validation_loss = evaluate(
        model=model,
        batches=loader,
        device="cpu",
    )

    assert math.isfinite(validation_loss)
    assert model.training
    assert all(parameter.grad is None for parameter in model.parameters())


def test_train_runs_accumulated_steps_with_scheduling_and_evaluation() -> None:
    torch.manual_seed(42)
    model = _tiny_model()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-2,
        weight_decay=0.0,
    )
    input_ids = torch.tensor(
        [
            [0, 4, 1, 4, 1, 4],
            [2, 4, 3, 4, 3, 4],
        ]
    )
    targets = torch.tensor(
        [
            [4, 1, 4, 1, 4, 1],
            [4, 3, 4, 3, 4, 3],
        ]
    )
    loader = DataLoader(
        TensorDataset(input_ids, targets),
        batch_size=2,
        shuffle=False,
    )
    embedding_before = model.token_embedding.weight.detach().clone()
    config = TrainingConfig(
        max_steps=3,
        grad_accum_steps=2,
        warmup_steps=1,
        max_learning_rate=1e-2,
        min_learning_rate=1e-3,
        max_grad_norm=1.0,
        eval_interval=2,
        eval_batches=1,
    )

    history = train(
        model=model,
        optimizer=optimizer,
        train_batches=loader,
        config=config,
        device="cpu",
        validation_batches=loader,
    )

    assert [metrics.step for metrics in history] == [1, 2, 3]
    assert [metrics.learning_rate for metrics in history] == [
        pytest.approx(1e-3),
        pytest.approx(1e-2),
        pytest.approx(5.5e-3),
    ]
    assert history[0].validation_loss is None
    assert history[1].validation_loss is not None
    assert history[2].validation_loss is None
    assert all(math.isfinite(metrics.loss) for metrics in history)
    assert all(math.isfinite(metrics.gradient_norm) for metrics in history)
    assert not torch.equal(model.token_embedding.weight, embedding_before)
