from contextlib import nullcontext
from pathlib import Path
from unittest.mock import Mock

import pytest

from mini_agent.agent.agent import Agent
from mini_agent.agent.events import AgentEvent, ContextUsage, ContextWarning, ToolStarted
from mini_agent.cli.renderer import CLIRenderer
from mini_agent.context import ContextLevel, ContextSettings
from mini_agent.telemetry.telemetry import LLMCallMetrics
from tests.integration.fakes import FakeLLM, final_response, tool_call_response


@pytest.fixture(autouse=True)
def offline_tracing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("mini_agent.agent.agent.agent_trace", lambda: nullcontext(Mock()))
    monkeypatch.setattr(
        "mini_agent.agent.tool_execution.tool_trace", lambda name: nullcontext(Mock())
    )


@pytest.mark.integration
def test_each_request_reports_its_own_context(sample_project: Path, capsys) -> None:
    llm = FakeLLM(
        [
            (
                tool_call_response("read_file", {"path": "README.md"}),
                LLMCallMetrics(0.1, 1000, 100),
            ),
            (
                tool_call_response("read_file", {"path": "src/main.py"}, "call_2"),
                LLMCallMetrics(0.1, 0, 0, usage_available=False),
            ),
            (final_response("Done"), LLMCallMetrics(0.1, 500, 200)),
        ]
    )
    renderer = CLIRenderer()
    events: list[ContextUsage] = []

    def on_event(event: AgentEvent) -> None:
        renderer.render(event)
        if isinstance(event, ContextUsage):
            events.append(event)

    agent = Agent(llm, on_event=on_event, context_settings=ContextSettings(context_window=10000))

    assert agent.run("Read the project files") == "Done"
    renderer.render_metrics(agent.metrics)

    assert events == [
        ContextUsage(1, 1000, 10000, 10.0),
        ContextUsage(2, None, 10000, None),
        ContextUsage(3, 500, 10000, 5.0),
    ]
    assert agent.metrics.input_tokens_total == 1500
    assert agent.metrics.context_input_tokens == 500
    assert agent.metrics.context_utilization_percent == 5.0
    assert [call.input_tokens for call in agent.metrics.llm_calls] == [1000, 0, 500]
    output = capsys.readouterr().out
    assert "Context (last request, iteration 1): 1,000 / 10,000 tokens (10.0%)" in output
    assert "Context (last request, iteration 2): unknown" in output
    assert "Done\nContext (last request, iteration 3): 500 / 10,000 tokens (5.0%)" in output
    assert "Context utilization (last request): 5.0%" in output


@pytest.mark.integration
@pytest.mark.parametrize("completed_first_request", [False, True])
def test_failed_request_does_not_invent_context_usage(
    sample_project: Path, completed_first_request: bool
) -> None:
    llm = Mock()
    responses = []
    if completed_first_request:
        responses.append(
            (
                tool_call_response("read_file", {"path": "README.md"}),
                LLMCallMetrics(0.1, 1000, 100),
            )
        )
    responses.append(RuntimeError("Provider unavailable"))
    llm.complete_stream.side_effect = responses
    events: list[AgentEvent] = []
    agent = Agent(
        llm, on_event=events.append, context_settings=ContextSettings(context_window=10000)
    )
    original_messages = agent.messages.copy()

    with pytest.raises(RuntimeError, match="Provider unavailable"):
        agent.run("Read README.md")

    context_events = [event for event in events if isinstance(event, ContextUsage)]
    if completed_first_request:
        assert context_events == [ContextUsage(1, 1000, 10000, 10.0)]
        assert agent.metrics.context_utilization_percent == 10.0
    else:
        assert context_events == []
        assert agent.metrics.context_utilization_percent is None
    assert agent.messages == original_messages


@pytest.mark.integration
def test_context_callback_failure_propagates_after_recording_request() -> None:
    llm = FakeLLM([(final_response("Done"), LLMCallMetrics(0.1, 1000, 100))])

    def on_event(event: AgentEvent) -> None:
        if isinstance(event, ContextUsage):
            raise ValueError("Renderer failed")

    agent = Agent(llm, on_event=on_event, context_settings=ContextSettings(context_window=10000))

    with pytest.raises(ValueError, match="Renderer failed"):
        agent.run("A task")

    assert len(agent.metrics.llm_calls) == 1
    assert agent.metrics.context_utilization_percent == 10.0
    assert len(agent.messages) == 1


@pytest.mark.integration
def test_new_task_starts_unknown_and_clear_preserves_latest_metrics() -> None:
    llm = FakeLLM([(final_response("Done"), LLMCallMetrics(0.1, 1000, 100))])
    agent = Agent(llm, context_settings=ContextSettings(context_window=10000))
    agent.run("First task")
    first_metrics = agent.metrics

    agent.reset_conversation()
    assert agent.metrics is first_metrics
    assert agent.metrics.context_utilization_percent == 10.0

    llm.complete_stream = Mock(side_effect=RuntimeError("Provider unavailable"))
    with pytest.raises(RuntimeError, match="Provider unavailable"):
        agent.run("Second task")

    assert agent.metrics is not first_metrics
    assert agent.metrics.context_window == 10000
    assert agent.metrics.context_input_tokens is None
    assert agent.metrics.context_utilization_percent is None


@pytest.mark.integration
def test_warnings_escalate_rearm_and_preserve_level_across_unknown_usage(
    sample_project: Path,
) -> None:
    inputs = [799, 800, 820, None, 950, None, 970, 900, 950, 790, 800, 1100]
    responses = []
    for iteration, input_tokens in enumerate(inputs, start=1):
        response = (
            final_response("Done")
            if iteration == len(inputs)
            else tool_call_response("read_file", {"path": "README.md"}, f"call_{iteration}")
        )
        responses.append(
            (
                response,
                LLMCallMetrics(
                    0.1,
                    input_tokens if input_tokens is not None else 0,
                    20 if input_tokens is not None else 0,
                    usage_available=input_tokens is not None,
                ),
            )
        )
    llm = FakeLLM(responses)
    events: list[AgentEvent] = []
    settings = ContextSettings(context_window=1200, output_reserve=200)
    agent = Agent(llm, on_event=events.append, context_settings=settings)

    assert agent.run("Read the project") == "Done"

    warnings = [event for event in events if isinstance(event, ContextWarning)]
    assert warnings == [
        ContextWarning(2, ContextLevel.WARNING, 800, 1000, 80.0),
        ContextWarning(5, ContextLevel.CRITICAL, 950, 1000, 95.0),
        ContextWarning(9, ContextLevel.CRITICAL, 950, 1000, 95.0),
        ContextWarning(11, ContextLevel.WARNING, 800, 1000, 80.0),
        ContextWarning(12, ContextLevel.CRITICAL, 1100, 1000, 110.0),
    ]
    for index, event in enumerate(events):
        if isinstance(event, ContextWarning):
            assert isinstance(events[index - 1], ContextUsage)
            assert events[index - 1].iteration == event.iteration
    assert agent.metrics.context_state.remaining_input_tokens == -100
    assert agent.metrics.context_state.settings is settings
    assert all(call["max_output_tokens"] == 200 for call in llm.calls)


@pytest.mark.integration
def test_first_critical_measurement_emits_one_warning_before_tools(sample_project: Path) -> None:
    llm = FakeLLM(
        [
            (
                tool_call_response("read_file", {"path": "README.md"}),
                LLMCallMetrics(0.1, 950, 20),
            ),
            (final_response("Done"), LLMCallMetrics(0.1, 0, 0, usage_available=False)),
        ]
    )
    events: list[AgentEvent] = []
    agent = Agent(
        llm,
        on_event=events.append,
        context_settings=ContextSettings(context_window=1200, output_reserve=200),
    )

    agent.run("Read README.md")

    assert isinstance(events[0], ContextUsage)
    assert events[1] == ContextWarning(1, ContextLevel.CRITICAL, 950, 1000, 95.0)
    assert isinstance(events[2], ToolStarted)
    assert len([event for event in events if isinstance(event, ContextWarning)]) == 1
    assert agent.metrics.context_state.level == ContextLevel.UNKNOWN


@pytest.mark.integration
def test_warning_state_resets_each_task_with_shared_capacity_settings() -> None:
    llm = FakeLLM(
        [
            (final_response("First"), LLMCallMetrics(0.1, 950, 20)),
            (final_response("Second"), LLMCallMetrics(0.1, 950, 20)),
        ]
    )
    events: list[AgentEvent] = []
    settings = ContextSettings(context_window=1200, output_reserve=200)
    agent = Agent(llm, on_event=events.append, context_settings=settings)

    agent.run("First task")
    previous_metrics = agent.metrics
    agent.run("Second task")

    assert [event for event in events if isinstance(event, ContextWarning)] == [
        ContextWarning(1, ContextLevel.CRITICAL, 950, 1000, 95.0),
        ContextWarning(1, ContextLevel.CRITICAL, 950, 1000, 95.0),
    ]
    assert previous_metrics is not agent.metrics
    assert previous_metrics.context_settings is agent.metrics.context_settings is settings
    assert len(agent.messages) == 5


@pytest.mark.integration
@pytest.mark.parametrize("kind", ["text", "tool", "partial_tool"])
def test_output_limit_preserves_metrics_and_history_without_executing_tools(
    project: Path, kind: str
) -> None:
    response = (
        final_response("This answer is incomplete")
        if kind == "text"
        else tool_call_response("write_file", {"path": "never.txt", "content": "Never written"})
    )
    if kind == "partial_tool":
        response.tool_calls[0].function.arguments = '{"path":'
    llm = FakeLLM(
        [
            (final_response("Previous answer"), LLMCallMetrics(0.1, 100, 20)),
            (response, LLMCallMetrics(0.1, 950, 200, finish_reason="length")),
        ]
    )
    events: list[AgentEvent] = []
    approve = Mock(return_value=True)
    agent = Agent(
        llm,
        on_event=events.append,
        approve=approve,
        context_settings=ContextSettings(context_window=1200, output_reserve=200),
    )
    agent.run("Previous task")
    previous_messages = agent.messages.copy()
    events.clear()

    with pytest.raises(RuntimeError, match="Response incomplete: output token limit reached"):
        agent.run("Create a file")

    assert agent.messages == previous_messages
    assert len(agent.metrics.llm_calls) == 1
    assert agent.metrics.llm_calls[0].output_tokens == 200
    assert agent.metrics.llm_calls[0].finish_reason == "length"
    assert agent.metrics.context_input_tokens == 950
    assert agent.metrics.context_state.level == ContextLevel.CRITICAL
    assert any(isinstance(event, ContextUsage) for event in events)
    assert any(isinstance(event, ContextWarning) for event in events)
    assert not any(isinstance(event, ToolStarted) for event in events)
    assert agent.metrics.tool_calls == 0
    assert not (project / "never.txt").exists()
    approve.assert_not_called()


@pytest.mark.integration
def test_warning_callback_failure_propagates_before_tool_execution(project: Path) -> None:
    llm = FakeLLM(
        [
            (
                tool_call_response("write_file", {"path": "never.txt", "content": "Never written"}),
                LLMCallMetrics(0.1, 800, 20),
            ),
        ]
    )

    def on_event(event: AgentEvent) -> None:
        if isinstance(event, ContextWarning):
            raise ValueError("Capacity renderer failed")

    agent = Agent(
        llm,
        on_event=on_event,
        context_settings=ContextSettings(context_window=1200, output_reserve=200),
    )

    with pytest.raises(ValueError, match="Capacity renderer failed"):
        agent.run("Create a file")

    assert agent.metrics.context_state.level == ContextLevel.WARNING
    assert agent.metrics.tool_calls == 0
    assert not (project / "never.txt").exists()
