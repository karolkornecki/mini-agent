import json
from collections import deque
from typing import Any

from openai.types.chat import ChatCompletionMessage

from mini_agent.telemetry.telemetry import LLMCallMetrics


class FakeLLM:
    def __init__(
        self,
        responses: list[tuple[ChatCompletionMessage, Any]],
    ) -> None:
        self.responses = deque(responses)
        self.calls: list[dict[str, Any]] = []

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> tuple[ChatCompletionMessage, Any]:
        self.calls.append(
            {
                "messages": list(messages),
                "tools": tools,
            }
        )

        if not self.responses:
            raise AssertionError("No more fake LLM responses")

        return self.responses.popleft()


def tool_call_response(
    name: str,
    arguments: dict[str, Any],
    call_id: str = "call_1",
) -> ChatCompletionMessage:
    return ChatCompletionMessage.model_validate(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(arguments),
                    },
                }
            ],
        }
    )


def final_response(content: str) -> ChatCompletionMessage:
    return ChatCompletionMessage(
        role="assistant",
        content=content,
    )


def fake_metrics() -> LLMCallMetrics:
    return LLMCallMetrics(
        latency_seconds=0.1,
        input_tokens=100,
        output_tokens=20,
    )
