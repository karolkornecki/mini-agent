from unittest.mock import Mock

import pytest

import mini_agent.main as main_module


@pytest.fixture
def startup(monkeypatch: pytest.MonkeyPatch) -> dict[str, Mock]:
    for name in (
        "MODEL_CONTEXT_WINDOW",
        "MODEL_OUTPUT_RESERVE",
        "CONTEXT_WARNING_PERCENT",
        "CONTEXT_CRITICAL_PERCENT",
        "MAX_TOOL_OUTPUT_CHARS",
    ):
        monkeypatch.delenv(name, raising=False)
    mocks = {}
    for name in ("load_dotenv", "LLMClient", "CLIRenderer", "run_cli", "flush"):
        mocks[name] = Mock()
        monkeypatch.setattr(main_module, name, mocks[name])
    return mocks


@pytest.mark.parametrize(("configured", "expected"), [(None, 128000), ("32000", 32000)])
def test_startup_passes_context_window_to_agent(monkeypatch, startup, configured, expected) -> None:
    if configured is None:
        monkeypatch.delenv("MODEL_CONTEXT_WINDOW", raising=False)
    else:
        monkeypatch.setenv("MODEL_CONTEXT_WINDOW", configured)

    main_module.main()

    agent, renderer = startup["run_cli"].call_args.args
    assert agent.context_settings.context_window == expected
    assert agent.metrics.context_window == expected
    assert agent.metrics.context_settings is agent.context_settings
    assert agent.context_settings.max_tool_output_chars == 16_000
    assert agent.on_event == renderer.render
    startup["flush"].assert_called_once_with()


@pytest.mark.parametrize("configured", ["", "abc", "1.5", "0", "-1000"])
def test_invalid_context_window_fails_before_client_initialization(
    monkeypatch, startup, configured
) -> None:
    monkeypatch.setenv("MODEL_CONTEXT_WINDOW", configured)

    with pytest.raises(SystemExit, match="Invalid context configuration"):
        main_module.main()

    startup["LLMClient"].assert_not_called()
    startup["run_cli"].assert_not_called()


def test_startup_passes_reserve_and_thresholds(monkeypatch, startup) -> None:
    monkeypatch.setenv("MODEL_OUTPUT_RESERVE", "4000")
    monkeypatch.setenv("CONTEXT_WARNING_PERCENT", "60")
    monkeypatch.setenv("CONTEXT_CRITICAL_PERCENT", "85")
    monkeypatch.setenv("MAX_TOOL_OUTPUT_CHARS", "400")

    main_module.main()

    agent, _ = startup["run_cli"].call_args.args
    assert agent.context_settings.output_reserve == 4000
    assert agent.context_settings.warning_percent == 60
    assert agent.context_settings.critical_percent == 85
    assert agent.context_settings.input_capacity == 124000
    assert agent.context_settings.max_tool_output_chars == 400


@pytest.mark.parametrize(
    ("variable", "value"),
    [
        ("MODEL_OUTPUT_RESERVE", "abc"),
        ("MODEL_OUTPUT_RESERVE", "0"),
        ("MODEL_OUTPUT_RESERVE", "-1"),
        ("MODEL_OUTPUT_RESERVE", "128000"),
        ("MODEL_OUTPUT_RESERVE", "128001"),
        ("CONTEXT_WARNING_PERCENT", "80.5"),
        ("CONTEXT_WARNING_PERCENT", "0"),
        ("CONTEXT_WARNING_PERCENT", "95"),
        ("CONTEXT_CRITICAL_PERCENT", "abc"),
        ("CONTEXT_CRITICAL_PERCENT", "80"),
        ("CONTEXT_CRITICAL_PERCENT", "101"),
        ("MAX_TOOL_OUTPUT_CHARS", "abc"),
        ("MAX_TOOL_OUTPUT_CHARS", "23"),
    ],
)
def test_invalid_context_settings_stop_startup(monkeypatch, startup, variable, value) -> None:
    monkeypatch.setenv(variable, value)

    with pytest.raises(SystemExit, match="Invalid context configuration"):
        main_module.main()

    startup["LLMClient"].assert_not_called()
    startup["run_cli"].assert_not_called()
