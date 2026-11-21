from contextlib import nullcontext
from pathlib import Path
from unittest.mock import Mock

import pytest

from mini_agent.agent.agent import Agent
from mini_agent.agent.events import AgentEvent, ContextUsage
from mini_agent.cli.renderer import CLIRenderer
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

    agent = Agent(llm, on_event=on_event, context_window=10000)

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
    agent = Agent(llm, on_event=events.append, context_window=10000)
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

    agent = Agent(llm, on_event=on_event, context_window=10000)

    with pytest.raises(ValueError, match="Renderer failed"):
        agent.run("A task")

    assert len(agent.metrics.llm_calls) == 1
    assert agent.metrics.context_utilization_percent == 10.0
    assert len(agent.messages) == 1


@pytest.mark.integration
def test_new_task_starts_unknown_and_clear_preserves_latest_metrics() -> None:
    llm = FakeLLM([(final_response("Done"), LLMCallMetrics(0.1, 1000, 100))])
    agent = Agent(llm, context_window=10000)
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
