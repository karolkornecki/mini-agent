from unittest.mock import MagicMock, patch

import pytest

from mini_agent.telemetry.observability import (
    agent_trace,
    flush,
    llm_trace,
    tool_trace,
)


@pytest.fixture
def langfuse_client():
    with patch("mini_agent.telemetry.observability.get_client") as get_client:
        client = MagicMock()
        get_client.return_value = client
        yield client


def test_agent_trace(langfuse_client):
    with agent_trace():
        pass

    langfuse_client.start_as_current_observation.assert_called_once_with(
        name="agent.run",
        as_type="agent",
    )


def test_llm_trace(langfuse_client):
    with llm_trace("qwen/qwen3-coder"):
        pass

    langfuse_client.start_as_current_observation.assert_called_once_with(
        name="llm.complete",
        as_type="generation",
        model="qwen/qwen3-coder",
    )


def test_tool_trace(langfuse_client):
    with tool_trace("read_file"):
        pass

    langfuse_client.start_as_current_observation.assert_called_once_with(
        name="tool.read_file",
        as_type="tool",
    )


def test_flush(langfuse_client):
    flush()

    langfuse_client.flush.assert_called_once()


def test_agent_trace_fallback(langfuse_client):
    langfuse_client.start_as_current_observation.side_effect = RuntimeError(
        "Langfuse unavailable"
    )

    with agent_trace() as observation:
        observation.update(metadata={"iterations": 1})


def test_flush_failure_does_not_propagate(langfuse_client):
    langfuse_client.flush.side_effect = RuntimeError(
        "Langfuse unavailable"
    )

    flush()

def test_agent_exception_propagates():
    with pytest.raises(ValueError, match="Agent failed"):
        with agent_trace():
            raise ValueError("Agent failed")