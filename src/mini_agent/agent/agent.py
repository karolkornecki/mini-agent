from collections.abc import Callable
from typing import Any

from mini_agent.agent.events import AgentEvent, ContextUsage, TextDelta
from mini_agent.agent.tool_execution import execute_tool
from mini_agent.llm.client import LLMClient
from mini_agent.permissions.approval import request_approval
from mini_agent.telemetry.observability import agent_trace
from mini_agent.telemetry.telemetry import AgentMetrics
from mini_agent.tools.registry import get_tool_schemas

SYSTEM_PROMPT = """
You are a coding assistant operating in the current repository.

For questions about this repository, its architecture, code,
behavior, or purpose:
- Inspect relevant files using tools before answering.
- Never assume repository contents from the conversation alone
  when verification is needed.
- If you cannot verify something, explicitly say so.
- Do not invent files, functions, or project details.

Before modifying code, inspect the relevant files.
Never read secrets or environment files.
"""


class Agent:
    def __init__(
        self,
        llm: LLMClient,
        on_event: Callable[[AgentEvent], None] | None = None,
        *,
        approve: Callable[[str, dict[str, Any]], bool] = request_approval,
        context_window: int = 128_000,
    ) -> None:
        self.llm = llm
        self.on_event = on_event or (lambda event: None)
        self.approve = approve
        self.context_window = context_window
        self.metrics = AgentMetrics(context_window=self.context_window)
        self.messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
        ]

    def run(self, task: str) -> str:
        self.metrics = AgentMetrics(context_window=self.context_window)

        with agent_trace() as trace:
            try:
                return self._run(task)
            finally:
                trace.update(
                    metadata={
                        "iterations": self.metrics.iterations,
                        "llm_calls": len(self.metrics.llm_calls),
                        "tool_calls": self.metrics.tool_calls,
                        "tool_errors": self.metrics.tool_errors,
                    }
                )

    def _run(self, task: str) -> str:
        messages = self.messages.copy()
        messages.append({"role": "user", "content": task})

        for _ in range(150):
            self.metrics.iterations += 1

            response, metrics = self.llm.complete_stream(
                messages=messages,
                tools=get_tool_schemas(),
                on_text=lambda text: self.on_event(TextDelta(text)),
            )

            self.metrics.llm_calls.append(metrics)
            self.on_event(
                ContextUsage(
                    iteration=self.metrics.iterations,
                    input_tokens=self.metrics.context_input_tokens,
                    context_window=self.metrics.context_window,
                    utilization_percent=self.metrics.context_utilization_percent,
                )
            )
            messages.append(response.model_dump(exclude_none=True))

            if not response.tool_calls:
                self.messages = messages
                return response.content or ""

            for call in response.tool_calls:
                messages.append(
                    execute_tool(
                        call,
                        metrics=self.metrics,
                        on_event=self.on_event,
                        approve=self.approve,
                    )
                )

        raise RuntimeError("Max iterations exceeded")

    def reset_conversation(self) -> None:
        self.messages = self.messages[:1]
