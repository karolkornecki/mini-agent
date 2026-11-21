import json
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import Mock

import pytest
from openai.types.chat import ChatCompletionMessage

from mini_agent.agent.agent import Agent
from mini_agent.agent.events import AgentEvent, ToolFinished
from mini_agent.cli.renderer import CLIRenderer
from mini_agent.context import ContextSettings
from mini_agent.telemetry.telemetry import LLMCallMetrics
from tests.integration.fakes import FakeLLM, fake_metrics, final_response, tool_call_response


@pytest.fixture(autouse=True)
def offline_tracing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("mini_agent.agent.agent.agent_trace", lambda: nullcontext(Mock()))
    monkeypatch.setattr(
        "mini_agent.agent.tool_execution.tool_trace", lambda name: nullcontext(Mock())
    )


def read_calls(*call_ids: str) -> ChatCompletionMessage:
    response = tool_call_response("read_file", {"path": "README.md"}, call_ids[0])
    for call_id in call_ids[1:]:
        call = tool_call_response("read_file", {"path": "README.md"}, call_id)
        response.tool_calls.extend(call.tool_calls)
    return response


@pytest.mark.integration
def test_tool_output_cap_preserves_call_ids_and_full_execution_events(project: Path) -> None:
    (project / "large.txt").write_text("é" * 64, encoding="utf-8")
    (project / "exact.txt").write_text("a" * 30, encoding="utf-8")
    response = tool_call_response("read_file", {"path": "large.txt"}, "large")
    exact = tool_call_response("read_file", {"path": "exact.txt"}, "exact")
    invalid = tool_call_response("unknown_tool_" * 4, {}, "error")
    denied = tool_call_response("write_file", {"path": "never.txt", "content": ""}, "denied")
    response.tool_calls.extend(exact.tool_calls + invalid.tool_calls + denied.tool_calls)
    llm = FakeLLM([(response, fake_metrics()), (final_response("Done"), fake_metrics())])
    events: list[AgentEvent] = []
    agent = Agent(
        llm,
        on_event=events.append,
        approve=lambda name, arguments: False,
        context_settings=ContextSettings(max_tool_output_chars=32),
    )

    agent.run("Read the files")

    messages = llm.calls[1]["messages"]
    results = messages[3:]
    assert results == [
        {"role": "tool", "tool_call_id": "large", "content": '"ééééééé\n[Tool result truncated]'},
        {"role": "tool", "tool_call_id": "exact", "content": '"' + "a" * 30 + '"'},
        {"role": "tool", "tool_call_id": "error", "content": "Tool err\n[Tool result truncated]"},
        {"role": "tool", "tool_call_id": "denied", "content": "User rej\n[Tool result truncated]"},
    ]
    assert [message["tool_call_id"] for message in results] == [
        call["id"] for call in messages[2]["tool_calls"]
    ]
    finished = [event for event in events if isinstance(event, ToolFinished)]
    assert [event.status for event in finished] == ["success", "success", "error", "denied"]
    assert json.loads(finished[0].output) == "é" * 64
    assert finished[2].output == "Tool error: Unknown tool: " + "unknown_tool_" * 4
    assert finished[3].output == "User rejected execution of write_file"
    assert agent.metrics.tool_calls == 4
    assert agent.metrics.tool_errors == 1
    assert agent.messages[3:-1] == results
    assert not (project / "never.txt").exists()


@pytest.mark.integration
def test_critical_requests_remove_one_whole_turn_at_a_time(sample_project: Path, capsys) -> None:
    critical = LLMCallMetrics(0.1, 950, 20)
    unknown = LLMCallMetrics(0.1, 0, 0, usage_available=False)
    llm = FakeLLM(
        [
            (read_calls("old"), fake_metrics()),
            (final_response("Old answer"), fake_metrics()),
            (read_calls("middle_1", "middle_2"), fake_metrics()),
            (final_response("Middle answer"), fake_metrics()),
            (final_response("Recent answer"), fake_metrics()),
            (read_calls("active_1", "active_2"), critical),
            (read_calls("active_3"), critical),
            (read_calls("active_4"), unknown),
            (final_response("Active answer"), fake_metrics()),
        ]
    )
    agent = Agent(
        llm,
        on_event=CLIRenderer().render,
        context_settings=ContextSettings(context_window=1200, output_reserve=200),
    )
    system = agent.messages[0]
    for task in ("Old", "Middle", "Recent", "Active"):
        agent.run(task)

    before, after_first, after_second, after_unknown = [call["messages"] for call in llm.calls[5:]]
    assert [message["content"] for message in before if message["role"] == "user"] == [
        "Old",
        "Middle",
        "Recent",
        "Active",
    ]
    assert after_first == [system, *before[5:], *after_first[-3:]]
    assert after_second == [system, *after_first[6:], *after_second[-2:]]
    assert after_unknown == [*after_second, *after_unknown[-2:]]
    for messages, expected_batches in (
        (after_first, [("middle_1", "middle_2"), ("active_1", "active_2")]),
        (after_second, [("active_1", "active_2"), ("active_3",)]),
        (after_unknown, [("active_1", "active_2"), ("active_3",), ("active_4",)]),
    ):
        batches = []
        for index, message in enumerate(messages):
            if message.get("tool_calls"):
                ids = tuple(call["id"] for call in message["tool_calls"])
                results = messages[index + 1 : index + 1 + len(ids)]
                assert all(result["role"] == "tool" for result in results)
                assert tuple(result["tool_call_id"] for result in results) == ids
                batches.append(ids)
        assert batches == expected_batches
        assert sum(message["role"] == "tool" for message in messages) == sum(
            len(ids) for ids in expected_batches
        )
    assert agent.messages == [*after_unknown, {"role": "assistant", "content": "Active answer"}]
    assert capsys.readouterr().out.count("Context: removed 1 older turn\n") == 2


@pytest.mark.integration
def test_pending_prune_survives_task_failure_and_is_consumed_on_success() -> None:
    llm = FakeLLM(
        [
            (final_response("Old answer"), LLMCallMetrics(0.1, 950, 20)),
            (final_response("Incomplete"), LLMCallMetrics(0.1, 950, 200, finish_reason="length")),
            (final_response("Retry answer"), LLMCallMetrics(0.1, 800, 20)),
            (final_response("Follow-up answer"), fake_metrics()),
        ]
    )
    agent = Agent(llm, context_settings=ContextSettings(context_window=1200, output_reserve=200))
    agent.run("Old task")
    saved = agent.messages.copy()

    with pytest.raises(RuntimeError, match="Response incomplete"):
        agent.run("Failed task")
    assert agent.messages == saved

    agent.run("Retry task")
    agent.run("Follow-up task")
    assert llm.calls[1]["messages"] == [saved[0], {"role": "user", "content": "Failed task"}]
    assert llm.calls[2]["messages"] == [saved[0], {"role": "user", "content": "Retry task"}]
    assert llm.calls[3]["messages"] == [
        saved[0],
        {"role": "user", "content": "Retry task"},
        {"role": "assistant", "content": "Retry answer"},
        {"role": "user", "content": "Follow-up task"},
    ]
