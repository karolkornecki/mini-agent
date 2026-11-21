import os
from collections.abc import Callable
from time import perf_counter
from typing import Any

from openai import OpenAI
from openai.types.chat import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageFunctionToolCall,
    Function,
)

from mini_agent.telemetry.observability import llm_trace
from mini_agent.telemetry.telemetry import LLMCallMetrics


class LLMClient:
    def __init__(self) -> None:
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
        )
        self.model = os.getenv(
            "MODEL",
            "qwen/qwen3-coder",
        )

    def complete_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        on_text: Callable[[str], None],
        *,
        max_output_tokens: int,
    ) -> tuple[ChatCompletionMessage, LLMCallMetrics]:
        with llm_trace(self.model) as generation:
            start = perf_counter()

            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                stream=True,
                stream_options={"include_usage": True},
                max_completion_tokens=max_output_tokens,
            )

            content_parts: list[str] = []
            tool_calls: dict[int, dict[str, Any]] = {}
            usage = None
            finish_reason = None

            for chunk in stream:
                if chunk.usage is not None:
                    usage = chunk.usage

                if not chunk.choices:
                    continue

                choice = chunk.choices[0]
                if choice.finish_reason is not None:
                    finish_reason = choice.finish_reason
                delta = choice.delta

                if delta.content:
                    content_parts.append(delta.content)
                    on_text(delta.content)

                for tool_delta in delta.tool_calls or []:
                    index = tool_delta.index

                    if index not in tool_calls:
                        tool_calls[index] = {
                            "id": "",
                            "name": "",
                            "arguments": "",
                        }

                    call = tool_calls[index]

                    if tool_delta.id:
                        call["id"] += tool_delta.id

                    if tool_delta.function:
                        if tool_delta.function.name:
                            call["name"] += tool_delta.function.name
                        if tool_delta.function.arguments:
                            call["arguments"] += tool_delta.function.arguments

            reconstructed_calls = [
                ChatCompletionMessageFunctionToolCall(
                    id=call["id"],
                    type="function",
                    function=Function(
                        name=call["name"],
                        arguments=call["arguments"],
                    ),
                )
                for _, call in sorted(tool_calls.items())
            ]

            response = ChatCompletionMessage(
                role="assistant",
                content="".join(content_parts) or None,
                tool_calls=reconstructed_calls or None,
            )

            latency = perf_counter() - start

            cached = 0
            if usage and usage.prompt_tokens_details:
                cached = usage.prompt_tokens_details.cached_tokens or 0

            metrics = LLMCallMetrics(
                latency_seconds=latency,
                input_tokens=usage.prompt_tokens if usage else 0,
                output_tokens=usage.completion_tokens if usage else 0,
                cached_tokens=cached,
                usage_available=usage is not None,
                finish_reason=finish_reason,
            )

            generation.update(
                usage_details={
                    "input": metrics.input_tokens,
                    "output": metrics.output_tokens,
                    "input_cached_tokens": metrics.cached_tokens,
                },
                metadata={
                    "latency_seconds": latency,
                    "tool_calls_requested": len(reconstructed_calls),
                    "finish_reason": finish_reason,
                },
            )

            return response, metrics
