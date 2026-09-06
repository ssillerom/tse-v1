import pytest

from training.scheduler import get_learning_rate, get_wsd_learning_rate


@pytest.mark.parametrize(
    ("warmup", "expected"), [(0, [1.0, 1.0, 1.0, 0.0]), (3, [0.0, 1 / 3, 2 / 3, 0.0])]
)
def test_wsd_with_no_decay_steps_preserves_warmup_and_stable_phase(warmup, expected):
    rates = [get_wsd_learning_rate(step, warmup, 3, 3, 1.0) for step in range(4)]
    assert rates == pytest.approx(expected)


@pytest.mark.parametrize(
    ("step", "expected_learning_rate"),
    [
        (0, 0.1),
        (1, 0.55),
        (2, 1.0),
        (6, 0.55),
        (10, 0.1),
        (12, 0.1),
    ],
)
def test_learning_rate_uses_linear_warmup_then_cosine_decay(
    step: int,
    expected_learning_rate: float,
) -> None:
    learning_rate = get_learning_rate(
        step=step,
        warmup_steps=2,
        max_steps=10,
        max_learning_rate=1.0,
        min_learning_rate=0.1,
    )

    assert learning_rate == pytest.approx(expected_learning_rate)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"step": -1}, "step must be a non-negative integer"),
        ({"warmup_steps": -1}, "warmup_steps must be a non-negative integer"),
        ({"max_steps": 0}, "max_steps must be a positive integer"),
        ({"warmup_steps": 11}, "warmup_steps cannot exceed max_steps"),
        ({"max_learning_rate": float("inf")}, "max_learning_rate must be finite and positive"),
        ({"min_learning_rate": -0.1}, "min_learning_rate must be finite and non-negative"),
        (
            {"max_learning_rate": 0.1, "min_learning_rate": 0.2},
            "min_learning_rate cannot exceed max_learning_rate",
        ),
    ],
)
def test_learning_rate_rejects_invalid_schedules(
    arguments: dict[str, int | float],
    message: str,
) -> None:
    valid_arguments: dict[str, int | float] = {
        "step": 0,
        "warmup_steps": 2,
        "max_steps": 10,
        "max_learning_rate": 1.0,
        "min_learning_rate": 0.1,
    }
    valid_arguments.update(arguments)

    with pytest.raises(ValueError, match=message):
        get_learning_rate(**valid_arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("step", "expected_learning_rate"),
    [
        (0, 0.0),
        (1, 0.5),
        (2, 1.0),
        (5, 1.0),
        (6, 1.0),
        (8, 0.25),
        (9, 0.0),
        (10, 0.0),
        (12, 0.0),
    ],
)
def test_wsd_learning_rate_warms_up_stays_constant_and_decays_to_zero(
    step: int,
    expected_learning_rate: float,
) -> None:
    learning_rate = get_wsd_learning_rate(
        step=step,
        warmup_steps=2,
        decay_start_step=6,
        max_steps=10,
        max_learning_rate=1.0,
    )

    assert learning_rate == pytest.approx(expected_learning_rate)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"decay_start_step": -1}, "decay_start_step must be a non-negative integer"),
        (
            {"warmup_steps": 7, "decay_start_step": 6},
            "warmup_steps cannot exceed decay_start_step",
        ),
        (
            {"decay_start_step": 11},
            "decay_start_step cannot exceed max_steps",
        ),
    ],
)
def test_wsd_learning_rate_rejects_invalid_phase_boundaries(
    arguments: dict[str, int],
    message: str,
) -> None:
    valid_arguments = {
        "step": 0,
        "warmup_steps": 2,
        "decay_start_step": 6,
        "max_steps": 10,
        "max_learning_rate": 1.0,
    }
    valid_arguments.update(arguments)

    with pytest.raises(ValueError, match=message):
        get_wsd_learning_rate(**valid_arguments)
