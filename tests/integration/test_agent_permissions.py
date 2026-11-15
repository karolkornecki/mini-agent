from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import mini_agent.agent.agent as agent_module
from mini_agent.agent.agent import Agent
from mini_agent.permissions.permissions import Permission
from tests.integration.fakes import FakeLLM, fake_metrics, final_response, tool_call_response


@contextmanager
def fake_trace(*args: Any, **kwargs: Any) -> Iterator[Any]:
    yield SimpleNamespace(update=lambda **kwargs: None)


@pytest.mark.integration
def test_read_file_allowed_without_approval(
    sample_project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Avoid external Langfuse calls.
    monkeypatch.setattr(agent_module, "agent_trace", fake_trace)
    monkeypatch.setattr(agent_module, "tool_trace", fake_trace)
    monkeypatch.setattr(
        agent_module,
        "request_approval",
        lambda name, arguments: False,
    )

    # Fail if Agent unexpectedly asks for approval.
    def unexpected_approval(prompt: str) -> str:
        raise AssertionError(f"Unexpected approval: {prompt}")

    monkeypatch.setattr("builtins.input", unexpected_approval)

    fake_llm = FakeLLM(
        [
            (
                tool_call_response("read_file", {"path": "src/main.py"}),
                fake_metrics(),
            ),
            (final_response("File successfully read"), fake_metrics()),
        ]
    )

    agent = Agent(llm=fake_llm)

    result = agent.run("Read src/main.py")

    assert result == "File successfully read"
    assert agent.metrics.iterations == 2
    assert agent.metrics.tool_calls == 1
    assert agent.metrics.tool_errors == 0

    assert len(fake_llm.calls) == 2

    second_call_messages = fake_llm.calls[1]["messages"]

    tool_messages = [message for message in second_call_messages if message["role"] == "tool"]

    assert len(tool_messages) == 1
    assert tool_messages[0]["tool_call_id"] == "call_1"
    assert "class Agent:" in tool_messages[0]["content"]


@pytest.mark.integration
def test_write_file_approved(
    project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(agent_module, "agent_trace", fake_trace)
    monkeypatch.setattr(agent_module, "tool_trace", fake_trace)
    monkeypatch.setattr(
        agent_module,
        "request_approval",
        lambda name, arguments: True,
    )

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

    agent = Agent(llm=fake_llm)
    result = agent.run("Create hello.txt")

    assert result == "File created"
    assert (project / "hello.txt").read_text(
        encoding="utf-8",
    ) == "Hello from mini-agent!"

    assert agent.metrics.tool_calls == 1
    assert agent.metrics.tool_errors == 0


@pytest.mark.integration
def test_write_file_rejected(
    project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(agent_module, "agent_trace", fake_trace)
    monkeypatch.setattr(agent_module, "tool_trace", fake_trace)
    monkeypatch.setattr(
        agent_module,
        "request_approval",
        lambda name, arguments: False,
    )

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

    agent = Agent(llm=fake_llm)
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
    monkeypatch.setattr(agent_module, "agent_trace", fake_trace)
    monkeypatch.setattr(agent_module, "tool_trace", fake_trace)

    # Force DENY for an existing tool.
    monkeypatch.setattr(
        agent_module,
        "get_permission",
        lambda _: Permission.DENY,
    )

    def unexpected_approval(prompt: str) -> str:
        raise AssertionError(f"Unexpected approval: {prompt}")

    monkeypatch.setattr("builtins.input", unexpected_approval)

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

    agent = Agent(llm=fake_llm)
    result = agent.run("Create denied.txt")

    assert result == "Operation denied"
    assert not (project / "denied.txt").exists()

    assert agent.metrics.tool_calls == 1
    assert agent.metrics.tool_errors == 1

    tool_message = fake_llm.calls[1]["messages"][-1]

    assert tool_message["role"] == "tool"
    assert tool_message["tool_call_id"] == "call_1"
    assert "Tool denied" in tool_message["content"]


@pytest.mark.integration
def test_unknown_tool_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(agent_module, "agent_trace", fake_trace)
    monkeypatch.setattr(agent_module, "tool_trace", fake_trace)

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
