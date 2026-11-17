import json
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from pathlib import Path
from subprocess import TimeoutExpired
from typing import Any
from unittest.mock import Mock

import pytest
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageFunctionToolCall,
    Function,
)

import mini_agent.agent.tool_execution as execution_module
from mini_agent.agent.events import AgentEvent, ToolFinished, ToolStarted
from mini_agent.agent.tool_execution import ApprovalRejected, execute_tool, execute_with_permission
from mini_agent.permissions.permissions import Permission
from mini_agent.telemetry.telemetry import AgentMetrics


@pytest.fixture(autouse=True)
def observation(monkeypatch: pytest.MonkeyPatch) -> Mock:
    observation = Mock()

    @contextmanager
    def trace(name: str) -> Iterator[Mock]:
        yield observation

    monkeypatch.setattr(execution_module, "tool_trace", trace)
    return observation


def make_call(name: str, arguments: str) -> ChatCompletionMessageFunctionToolCall:
    return ChatCompletionMessageFunctionToolCall(
        id="call_1",
        type="function",
        function=Function(name=name, arguments=arguments),
    )


@pytest.mark.parametrize(
    ("name", "arguments", "permission", "approved", "status", "error_count", "output"),
    [
        ("read_file", {"path": "answer.txt"}, Permission.ALLOW, False, "success", 0, '"Cześć"'),
        (
            "write_file",
            {"path": "new.txt", "content": "hello"},
            Permission.ASK,
            False,
            "denied",
            0,
            "User rejected execution of write_file",
        ),
        (
            "write_file",
            {"path": "new.txt", "content": "hello"},
            Permission.DENY,
            True,
            "error",
            1,
            "Tool error: Tool denied: write_file",
        ),
        (
            "missing_tool",
            {},
            Permission.ALLOW,
            True,
            "error",
            1,
            "Tool error: Unknown tool: missing_tool",
        ),
    ],
)
def test_outcomes_keep_events_results_and_metrics_consistent(
    project: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    arguments: dict[str, Any],
    permission: Permission,
    approved: bool,
    status: str,
    error_count: int,
    output: str,
) -> None:
    (project / "answer.txt").write_text("Cześć", encoding="utf-8")
    monkeypatch.setattr(execution_module, "get_permission", lambda name: permission)
    metrics = AgentMetrics()
    events: list[AgentEvent] = []
    approve = Mock(return_value=approved)

    message = execute_tool(
        make_call(name, json.dumps(arguments)),
        metrics=metrics,
        on_event=events.append,
        approve=approve,
    )

    assert message == {"role": "tool", "tool_call_id": "call_1", "content": output}
    assert events == [
        ToolStarted("call_1", name, arguments),
        ToolFinished("call_1", name, status, output),
    ]
    assert metrics.tool_calls == 1
    assert metrics.tool_errors == error_count
    assert not (project / "new.txt").exists()
    if permission == Permission.ASK:
        approve.assert_called_once_with(name, arguments)
    else:
        approve.assert_not_called()


@pytest.mark.parametrize("arguments", ["{", "[]", "null"])
def test_invalid_arguments_finish_with_an_error_without_approval(arguments: str) -> None:
    metrics = AgentMetrics()
    events: list[AgentEvent] = []
    approve = Mock(side_effect=AssertionError("Unexpected approval"))

    message = execute_tool(
        make_call("write_file", arguments),
        metrics=metrics,
        on_event=events.append,
        approve=approve,
    )

    assert message["role"] == "tool"
    assert message["tool_call_id"] == "call_1"
    assert message["content"].startswith("Tool error: ")
    if arguments != "{":
        assert message["content"] == "Tool error: Arguments must be an object"
    assert events == [
        ToolStarted("call_1", "write_file", {}),
        ToolFinished("call_1", "write_file", "error", message["content"]),
    ]
    assert metrics.tool_calls == 1
    assert metrics.tool_errors == 1
    approve.assert_not_called()


@pytest.mark.parametrize(
    "error",
    [
        ValueError("bad value"),
        TypeError("bad type"),
        OSError("unavailable"),
        PermissionError("no access"),
        KeyError("missing"),
        TimeoutExpired(cmd=["git", "status"], timeout=30),
    ],
)
def test_handled_tool_failures_return_errors(
    monkeypatch: pytest.MonkeyPatch,
    observation: Mock,
    error: Exception,
) -> None:
    tool = replace(execution_module.REGISTRY["read_file"], execute=Mock(side_effect=error))
    monkeypatch.setitem(execution_module.REGISTRY, "read_file", tool)
    metrics = AgentMetrics()
    events: list[AgentEvent] = []

    message = execute_tool(
        make_call("read_file", '{"path": "answer.txt"}'),
        metrics=metrics,
        on_event=events.append,
        approve=Mock(side_effect=AssertionError("Unexpected approval")),
    )

    assert message == {"role": "tool", "tool_call_id": "call_1", "content": f"Tool error: {error}"}
    assert events == [
        ToolStarted("call_1", "read_file", {"path": "answer.txt"}),
        ToolFinished("call_1", "read_file", "error", message["content"]),
    ]
    assert metrics.tool_calls == 1
    assert metrics.tool_errors == 1
    observation.update.assert_called_once_with(level="ERROR", status_message=type(error).__name__)


@pytest.mark.parametrize("error", [RuntimeError("bug"), AssertionError("bug")])
def test_unhandled_tool_failures_propagate(
    monkeypatch: pytest.MonkeyPatch,
    observation: Mock,
    error: Exception,
) -> None:
    tool = replace(execution_module.REGISTRY["read_file"], execute=Mock(side_effect=error))
    monkeypatch.setitem(execution_module.REGISTRY, "read_file", tool)
    metrics = AgentMetrics()
    events: list[AgentEvent] = []

    with pytest.raises(type(error), match="bug"):
        execute_tool(
            make_call("read_file", '{"path": "answer.txt"}'),
            metrics=metrics,
            on_event=events.append,
            approve=Mock(),
        )

    assert events == [ToolStarted("call_1", "read_file", {"path": "answer.txt"})]
    assert metrics.tool_calls == 1
    assert metrics.tool_errors == 0
    observation.update.assert_called_once_with(level="ERROR", status_message=type(error).__name__)


@pytest.mark.parametrize("name", ["read_file", "write_file"])
def test_permitted_call_returns_the_callable_result(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    observation = Mock()
    trace = Mock(return_value=nullcontext(observation))
    monkeypatch.setattr(execution_module, "tool_trace", trace)
    execute = Mock(return_value=None)
    approve = Mock(return_value=True)
    arguments = {"path": "example.txt"}

    result = execute_with_permission(execute, name=name, arguments=arguments, approve=approve)

    assert result is None
    execute.assert_called_once_with(path="example.txt")
    trace.assert_called_once_with(name)
    observation.update.assert_not_called()
    if name == "write_file":
        approve.assert_called_once_with(name, arguments)
    else:
        approve.assert_not_called()


@pytest.mark.parametrize(
    ("name", "approved", "error", "message"),
    [
        ("write_file", False, ApprovalRejected, "User rejected execution of write_file"),
        ("unlisted_tool", True, PermissionError, "Tool denied: unlisted_tool"),
    ],
)
def test_blocked_call_never_executes_or_starts_a_trace(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    approved: bool,
    error: type[Exception],
    message: str,
) -> None:
    trace = Mock(side_effect=AssertionError("Unexpected tool trace"))
    monkeypatch.setattr(execution_module, "tool_trace", trace)
    execute = Mock(side_effect=AssertionError("Unexpected execution"))
    approve = Mock(return_value=approved)
    arguments = {"path": "example.txt"}

    with pytest.raises(error, match=message):
        execute_with_permission(execute, name=name, arguments=arguments, approve=approve)

    execute.assert_not_called()
    trace.assert_not_called()
    if name == "write_file":
        approve.assert_called_once_with(name, arguments)
    else:
        approve.assert_not_called()


@pytest.mark.parametrize("event_type", [ToolStarted, ToolFinished])
def test_event_callback_failures_propagate(
    project: Path,
    event_type: type[ToolStarted] | type[ToolFinished],
) -> None:
    (project / "answer.txt").write_text("answer", encoding="utf-8")
    metrics = AgentMetrics()

    def on_event(event: AgentEvent) -> None:
        if isinstance(event, event_type):
            raise ValueError("renderer failed")

    with pytest.raises(ValueError, match="renderer failed"):
        execute_tool(
            make_call("read_file", '{"path": "answer.txt"}'),
            metrics=metrics,
            on_event=on_event,
            approve=Mock(),
        )

    assert metrics.tool_calls == 1
    assert metrics.tool_errors == 0
