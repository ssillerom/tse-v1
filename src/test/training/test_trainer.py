import math

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from model.config import ModelConfig
from model.gpt import GPT
from training.trainer import TrainingConfig, evaluate, train, train_step


def _tiny_model(dropout: float = 0.0) -> GPT:
    return GPT(
        ModelConfig(
            vocab_size=8,
            d_model=16,
            n_layers=2,
            n_heads=4,
            max_seq_len=6,
            dropout=dropout,
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


def test_train_step_skips_a_microbatch_without_target_tokens() -> None:
    torch.manual_seed(42)
    accumulated_model = _tiny_model()
    valid_only_model = _tiny_model()
    valid_only_model.load_state_dict(accumulated_model.state_dict())
    accumulated_optimizer = torch.optim.SGD(accumulated_model.parameters(), lr=1e-2)
    valid_only_optimizer = torch.optim.SGD(valid_only_model.parameters(), lr=1e-2)
    ignored_batch = (
        torch.tensor([[0, 1, 2, 3, 4, 5]]),
        torch.full((1, 6), -100),
    )
    valid_batch = (
        torch.tensor([[1, 2, 3, 4, 5, 6]]),
        torch.tensor([[2, 3, 4, 5, 6, 7]]),
    )

    accumulated_loss, _ = train_step(
        model=accumulated_model,
        optimizer=accumulated_optimizer,
        microbatches=[ignored_batch, valid_batch],
        max_grad_norm=1e6,
        device="cpu",
    )
    valid_only_loss, _ = train_step(
        model=valid_only_model,
        optimizer=valid_only_optimizer,
        microbatches=[valid_batch],
        max_grad_norm=1e6,
        device="cpu",
    )

    assert accumulated_loss == pytest.approx(valid_only_loss)
    for accumulated_parameter, valid_only_parameter in zip(
        accumulated_model.parameters(),
        valid_only_model.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(accumulated_parameter, valid_only_parameter)


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


def test_evaluate_skips_batches_without_target_tokens() -> None:
    torch.manual_seed(42)
    model = _tiny_model()
    ignored_batch = (
        torch.tensor([[0, 1, 2, 3, 4, 5]]),
        torch.full((1, 6), -100),
    )
    valid_batch = (
        torch.tensor([[1, 2, 3, 4, 5, 6]]),
        torch.tensor([[2, 3, 4, 5, 6, 7]]),
    )

    mixed_loss = evaluate(
        model=model,
        batches=[ignored_batch, valid_batch],
        device="cpu",
    )
    valid_only_loss = evaluate(
        model=model,
        batches=[valid_batch],
        device="cpu",
    )

    assert mixed_loss == pytest.approx(valid_only_loss)


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


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "mps",
            marks=pytest.mark.skipif(
                not torch.backends.mps.is_available(),
                reason="MPS is not available",
            ),
        ),
    ],
)
def test_train_resume_matches_an_uninterrupted_deterministic_run(
    device: str,
) -> None:
    torch.manual_seed(42)
    if device == "mps":
        torch.mps.manual_seed(42)
    uninterrupted_model = _tiny_model(dropout=0.2).to(device)
    resumed_model = _tiny_model(dropout=0.2).to(device)
    resumed_model.load_state_dict(uninterrupted_model.state_dict())
    uninterrupted_optimizer = torch.optim.AdamW(
        uninterrupted_model.parameters(),
        lr=1e-2,
        weight_decay=0.0,
    )
    resumed_optimizer = torch.optim.AdamW(
        resumed_model.parameters(),
        lr=1e-2,
        weight_decay=0.0,
    )
    input_ids = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5, 6],
            [2, 3, 4, 5, 6, 7],
        ]
    )
    targets = torch.tensor(
        [
            [1, 2, 3, 4, 5, 6],
            [2, 3, 4, 5, 6, 7],
            [3, 4, 5, 6, 7, 0],
        ]
    )
    loader = DataLoader(
        TensorDataset(input_ids, targets),
        batch_size=1,
        shuffle=False,
    )
    config = TrainingConfig(
        max_steps=4,
        grad_accum_steps=2,
        warmup_steps=1,
        max_learning_rate=1e-2,
        min_learning_rate=1e-3,
        max_grad_norm=1.0,
    )
    initial_training_rng_state = torch.random.get_rng_state()
    initial_mps_rng_state = torch.mps.get_rng_state() if device == "mps" else None

    train(
        model=uninterrupted_model,
        optimizer=uninterrupted_optimizer,
        train_batches=loader,
        config=config,
        device=device,
    )
    torch.random.set_rng_state(initial_training_rng_state)
    if initial_mps_rng_state is not None:
        torch.mps.set_rng_state(initial_mps_rng_state)
    first_segment = train(
        model=resumed_model,
        optimizer=resumed_optimizer,
        train_batches=loader,
        config=config,
        device=device,
        end_step=2,
    )
    checkpoint_rng_state = torch.random.get_rng_state()
    checkpoint_mps_rng_state = torch.mps.get_rng_state() if device == "mps" else None
    torch.rand(10)
    if device == "mps":
        torch.rand(10, device="mps")
    torch.random.set_rng_state(checkpoint_rng_state)
    if checkpoint_mps_rng_state is not None:
        torch.mps.set_rng_state(checkpoint_mps_rng_state)
    second_segment = train(
        model=resumed_model,
        optimizer=resumed_optimizer,
        train_batches=loader,
        config=config,
        device=device,
        start_step=2,
    )

    assert [metrics.step for metrics in first_segment] == [1, 2]
    assert [metrics.step for metrics in second_segment] == [3, 4]
    for uninterrupted_parameter, resumed_parameter in zip(
        uninterrupted_model.parameters(),
        resumed_model.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(
            resumed_parameter.detach().cpu(),
            uninterrupted_parameter.detach().cpu(),
            rtol=0.0,
            atol=0.0,
        )


def test_train_rejects_a_one_shot_batch_iterator() -> None:
    model = _tiny_model()
    optimizer = torch.optim.AdamW(model.parameters())
    batch_iterator = iter(
        [
            (
                torch.tensor([[0, 1, 2, 3, 4, 5]]),
                torch.tensor([[1, 2, 3, 4, 5, 6]]),
            )
        ]
    )

    with pytest.raises(
        ValueError,
        match="train_batches must be reiterable",
    ):
        train(
            model=model,
            optimizer=optimizer,
            train_batches=batch_iterator,
            config=TrainingConfig(max_steps=1),
            device="cpu",
        )


def test_training_config_rejects_eval_batches_without_an_interval() -> None:
    with pytest.raises(
        ValueError,
        match="eval_batches requires eval_interval",
    ):
        TrainingConfig(
            max_steps=1,
            eval_batches=1,
        )
