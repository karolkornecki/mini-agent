import os
from time import perf_counter
from typing import Any

from openai import OpenAI

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

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> tuple[Any, LLMCallMetrics]:
        start = perf_counter()

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
        )

        latency = perf_counter() - start
        usage = response.usage

        cached = 0
        if usage and usage.prompt_tokens_details:
            cached = usage.prompt_tokens_details.cached_tokens or 0

        metrics = LLMCallMetrics(
            latency_seconds=latency,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            cached_tokens=cached,
        )

        return response.choices[0].message, metrics
