import pytest

from mini_agent.context import ContextLevel, ContextSettings, ContextState


@pytest.mark.parametrize(
    ("input_tokens", "expected_level"),
    [
        (None, ContextLevel.UNKNOWN),
        (0, ContextLevel.NORMAL),
        (799, ContextLevel.NORMAL),
        (800, ContextLevel.WARNING),
        (801, ContextLevel.WARNING),
        (949, ContextLevel.WARNING),
        (950, ContextLevel.CRITICAL),
        (951, ContextLevel.CRITICAL),
        (1000, ContextLevel.CRITICAL),
        (1100, ContextLevel.CRITICAL),
    ],
)
def test_levels_use_input_capacity_and_inclusive_thresholds(input_tokens, expected_level) -> None:
    settings = ContextSettings(context_window=1200, output_reserve=200)

    assert ContextState(settings, input_tokens).level == expected_level


def test_state_keeps_full_context_and_input_capacity_utilization_separate() -> None:
    state = ContextState(ContextSettings(context_window=2000, output_reserve=1000), 800)

    assert state.utilization_percent == 40.0
    assert state.input_utilization_percent == 80.0
    assert state.remaining_input_tokens == 200
    assert state.level == ContextLevel.WARNING


def test_exceeded_input_capacity_retains_negative_headroom() -> None:
    state = ContextState(ContextSettings(context_window=1200, output_reserve=200), 1100)

    assert state.input_utilization_percent == pytest.approx(110.0)
    assert state.remaining_input_tokens == -100
    assert state.level == ContextLevel.CRITICAL


def test_unknown_state_has_no_numeric_measurement() -> None:
    state = ContextState(ContextSettings(), None)

    assert state.utilization_percent is None
    assert state.input_utilization_percent is None
    assert state.remaining_input_tokens is None
    assert state.level == ContextLevel.UNKNOWN


@pytest.mark.parametrize(
    ("input_tokens", "expected_level"),
    [
        (499, ContextLevel.NORMAL),
        (500, ContextLevel.WARNING),
        (749, ContextLevel.WARNING),
        (750, ContextLevel.CRITICAL),
    ],
)
def test_custom_thresholds(input_tokens, expected_level) -> None:
    settings = ContextSettings(
        context_window=1200, output_reserve=200, warning_percent=50, critical_percent=75
    )

    assert ContextState(settings, input_tokens).level == expected_level


@pytest.mark.parametrize(
    "settings",
    [
        {"context_window": 0},
        {"context_window": -1},
        {"output_reserve": 0},
        {"output_reserve": -1},
        {"output_reserve": 128000},
        {"output_reserve": 128001},
        {"warning_percent": 0},
        {"warning_percent": 95},
        {"critical_percent": 80},
        {"critical_percent": 101},
    ],
)
def test_invalid_settings_are_rejected(settings) -> None:
    with pytest.raises(ValueError):
        ContextSettings(**settings)
