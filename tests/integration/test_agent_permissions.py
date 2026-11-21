import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

import mini_agent.agent.agent as agent_module
import mini_agent.agent.tool_execution as execution_module
from mini_agent.agent.agent import Agent
from mini_agent.agent.events import AgentEvent, ContextUsage, TextDelta, ToolFinished, ToolStarted
from mini_agent.context import ContextSettings
from mini_agent.permissions.permissions import Permission
from tests.integration.fakes import FakeLLM, fake_metrics, final_response, tool_call_response


@contextmanager
def fake_trace(*args: Any, **kwargs: Any) -> Iterator[Any]:
    yield SimpleNamespace(update=lambda **kwargs: None)


@pytest.fixture(autouse=True)
def offline_tracing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(agent_module, "agent_trace", fake_trace)
    monkeypatch.setattr(execution_module, "tool_trace", fake_trace)


@pytest.mark.integration
def test_read_file_allowed_without_approval(
    sample_project: Path,
) -> None:
    approve = Mock(side_effect=AssertionError("Unexpected approval"))

    fake_llm = FakeLLM(
        [
            (
                tool_call_response("read_file", {"path": "src/main.py"}),
                fake_metrics(),
            ),
            (final_response("File successfully read"), fake_metrics()),
        ]
    )

    agent = Agent(llm=fake_llm, approve=approve)

    result = agent.run("Read src/main.py")

    assert result == "File successfully read"
    assert agent.metrics.iterations == 2
    assert agent.metrics.tool_calls == 1
    assert agent.metrics.tool_errors == 0
    approve.assert_not_called()

    assert len(fake_llm.calls) == 2

    second_call_messages = fake_llm.calls[1]["messages"]

    tool_messages = [message for message in second_call_messages if message["role"] == "tool"]

    assert len(tool_messages) == 1
    assert tool_messages[0]["tool_call_id"] == "call_1"
    assert "class Agent:" in tool_messages[0]["content"]


@pytest.mark.integration
def test_write_file_approved(
    project: Path,
) -> None:
    approve = Mock(return_value=True)

    fake_llm = FakeLLM(
        [
            (
                tool_call_response(
                    "write_file",
                    {
                        "path": "hello.txt",
                        "content": "Hello from mini-agent!",
                    },
                    call_id="call_write_1",
                ),
                fake_metrics(),
            ),
            (final_response("File created"), fake_metrics()),
        ]
    )

    agent = Agent(llm=fake_llm, approve=approve)
    result = agent.run("Create hello.txt")

    assert result == "File created"
    assert (project / "hello.txt").read_text(
        encoding="utf-8",
    ) == "Hello from mini-agent!"

    assert agent.metrics.tool_calls == 1
    assert agent.metrics.tool_errors == 0
    approve.assert_called_once_with(
        "write_file", {"path": "hello.txt", "content": "Hello from mini-agent!"}
    )


@pytest.mark.integration
def test_write_file_rejected(
    project: Path,
) -> None:
    fake_llm = FakeLLM(
        [
            (
                tool_call_response(
                    "write_file",
                    {
                        "path": "hello.txt",
                        "content": "This should not be written",
                    },
                    call_id="call_write_2",
                ),
                fake_metrics(),
            ),
            (final_response("Write rejected"), fake_metrics()),
        ]
    )

    agent = Agent(llm=fake_llm, approve=lambda name, arguments: False)
    result = agent.run("Create hello.txt")

    assert result == "Write rejected"
    assert not (project / "hello.txt").exists()

    assert agent.metrics.tool_calls == 1
    assert agent.metrics.tool_errors == 0

    tool_messages = [
        message for message in fake_llm.calls[1]["messages"] if message["role"] == "tool"
    ]

    assert len(tool_messages) == 1
    assert "User rejected execution" in tool_messages[0]["content"]


@pytest.mark.integration
def test_tool_denied(
    project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Force DENY for an existing tool.
    monkeypatch.setattr(
        execution_module,
        "get_permission",
        lambda _: Permission.DENY,
    )

    approve = Mock(side_effect=AssertionError("Unexpected approval"))

    fake_llm = FakeLLM(
        [
            (
                tool_call_response(
                    "write_file",
                    {"path": "denied.txt", "content": "Secret"},
                ),
                fake_metrics(),
            ),
            (final_response("Operation denied"), fake_metrics()),
        ]
    )

    agent = Agent(llm=fake_llm, approve=approve)
    result = agent.run("Create denied.txt")

    assert result == "Operation denied"
    assert not (project / "denied.txt").exists()

    assert agent.metrics.tool_calls == 1
    assert agent.metrics.tool_errors == 1
    approve.assert_not_called()

    tool_message = fake_llm.calls[1]["messages"][-1]

    assert tool_message["role"] == "tool"
    assert tool_message["tool_call_id"] == "call_1"
    assert "Tool denied" in tool_message["content"]


@pytest.mark.integration
def test_unknown_tool_is_rejected() -> None:
    fake_llm = FakeLLM(
        [
            (
                tool_call_response(
                    "delete_everything",
                    {"path": "."},
                ),
                fake_metrics(),
            ),
            (final_response("Unknown tool rejected"), fake_metrics()),
        ]
    )

    agent = Agent(llm=fake_llm)
    result = agent.run("Execute unknown tool")

    assert result == "Unknown tool rejected"
    assert agent.metrics.tool_calls == 1
    assert agent.metrics.tool_errors == 1

    tool_message = fake_llm.calls[1]["messages"][-1]

    assert "Unknown tool: delete_everything" in (tool_message["content"])


@pytest.mark.integration
def test_timeout_returns_result_and_continues_with_remaining_calls(
    sample_project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timeout = subprocess.TimeoutExpired(cmd=["git", "status", "--short"], timeout=30)
    run = Mock(side_effect=timeout)
    monkeypatch.setattr(subprocess, "run", run)
    approve = Mock(return_value=True)

    response = tool_call_response("run_command", {"command": "git_status"}, "call_timeout")
    read = tool_call_response("read_file", {"path": "README.md"}, "call_read")
    assert response.tool_calls is not None
    assert read.tool_calls is not None
    response.tool_calls.extend(read.tool_calls)
    fake_llm = FakeLLM(
        [
            (response, fake_metrics()),
            (final_response("Recovered"), fake_metrics()),
        ]
    )
    events: list[AgentEvent] = []
    agent = Agent(fake_llm, on_event=events.append, approve=approve)

    assert agent.run("Run git status and read README.md") == "Recovered"

    tool_messages = fake_llm.calls[1]["messages"][-2:]
    assert tool_messages == [
        {"role": "tool", "tool_call_id": "call_timeout", "content": f"Tool error: {timeout}"},
        {"role": "tool", "tool_call_id": "call_read", "content": '"# Mini Agent\\n"'},
    ]
    assert events == [
        ContextUsage(1, 100, 128000, 0.078125),
        ToolStarted("call_timeout", "run_command", {"command": "git_status"}),
        ToolFinished("call_timeout", "run_command", "error", tool_messages[0]["content"]),
        ToolStarted("call_read", "read_file", {"path": "README.md"}),
        ToolFinished("call_read", "read_file", "success", tool_messages[1]["content"]),
        TextDelta("Recovered"),
        ContextUsage(2, 100, 128000, 0.078125),
    ]
    assert agent.metrics.iterations == 2
    assert agent.metrics.tool_calls == 2
    assert agent.metrics.tool_errors == 1
    assert agent.messages == fake_llm.calls[1]["messages"] + [
        {"role": "assistant", "content": "Recovered"}
    ]
    approve.assert_called_once_with("run_command", {"command": "git_status"})
    run.assert_called_once()


@pytest.mark.integration
def test_execution_uses_fresh_metrics_for_each_task(sample_project: Path) -> None:
    fake_llm = FakeLLM(
        [
            (tool_call_response("read_file", {"path": "README.md"}), fake_metrics()),
            (final_response("Read"), fake_metrics()),
            (final_response("Next"), fake_metrics()),
        ]
    )
    agent = Agent(
        fake_llm, context_settings=ContextSettings(context_window=1000, output_reserve=100)
    )

    agent.run("Read README.md")
    first_metrics = agent.metrics
    agent.run("Next task")

    assert first_metrics.tool_calls == 1
    assert agent.metrics is not first_metrics
    assert agent.metrics.tool_calls == 0
    assert agent.metrics.tool_errors == 0
    assert first_metrics.context_window == agent.metrics.context_window == 1000
    assert first_metrics.input_tokens_total == 200
    assert agent.metrics.input_tokens_total == 100
    assert agent.metrics.context_utilization_percent == 10.0
