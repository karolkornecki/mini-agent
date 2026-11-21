from copy import deepcopy
from unittest.mock import Mock, patch

from mini_agent.agent.agent import Agent
from mini_agent.telemetry.telemetry import LLMCallMetrics


def test_conversation_preserves_history() -> None:
    llm = Mock()
    captured_messages = []

    def complete_stream(*, messages, tools, on_text):
        captured_messages.append(deepcopy(messages))

        response = Mock()
        response.content = "Done"
        response.tool_calls = None
        response.model_dump.return_value = {
            "role": "assistant",
            "content": "Done",
        }
        on_text(response.content)
        return response, LLMCallMetrics(0.1, 100, 20)

    llm.complete_stream.side_effect = complete_stream

    agent = Agent(llm)

    with patch("mini_agent.agent.agent.agent_trace"):
        agent.run("First task")
        agent.run("Second task")

    assert llm.complete_stream.call_count == 2

    assert [m["role"] for m in captured_messages[0]] == [
        "system",
        "user",
    ]

    assert [m["role"] for m in captured_messages[1]] == [
        "system",
        "user",
        "assistant",
        "user",
    ]

    assert captured_messages[1][1]["content"] == "First task"
    assert captured_messages[1][2]["content"] == "Done"
    assert captured_messages[1][3]["content"] == "Second task"

    assert [m["role"] for m in agent.messages] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
    ]


def test_reset_conversation() -> None:
    agent = Agent(Mock())
    agent.messages.append(
        {
            "role": "user",
            "content": "Previous task",
        }
    )

    agent.reset_conversation()

    assert len(agent.messages) == 1
    assert agent.messages[0]["role"] == "system"
