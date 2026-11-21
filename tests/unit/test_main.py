from unittest.mock import Mock

import pytest

import mini_agent.main as main_module


@pytest.fixture
def startup(monkeypatch: pytest.MonkeyPatch) -> dict[str, Mock]:
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
    assert agent.context_window == expected
    assert agent.metrics.context_window == expected
    assert agent.on_event == renderer.render
    startup["flush"].assert_called_once_with()


@pytest.mark.parametrize("configured", ["", "abc", "1.5", "0", "-1000"])
def test_invalid_context_window_fails_before_client_initialization(
    monkeypatch, startup, configured
) -> None:
    monkeypatch.setenv("MODEL_CONTEXT_WINDOW", configured)

    with pytest.raises(SystemExit, match="MODEL_CONTEXT_WINDOW must be a positive integer"):
        main_module.main()

    startup["LLMClient"].assert_not_called()
    startup["run_cli"].assert_not_called()
