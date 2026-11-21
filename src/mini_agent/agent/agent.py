from collections.abc import Callable
from typing import Any

from mini_agent.agent.events import AgentEvent, ContextUsage, ContextWarning, TextDelta
from mini_agent.agent.tool_execution import execute_tool
from mini_agent.context import ContextLevel, ContextSettings
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
        context_settings: ContextSettings | None = None,
    ) -> None:
        self.llm = llm
        self.on_event = on_event or (lambda event: None)
        self.approve = approve
        self.context_settings = (
            context_settings if context_settings is not None else ContextSettings()
        )
        self.metrics = AgentMetrics(context_settings=self.context_settings)
        self.messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
        ]

    def run(self, task: str) -> str:
        self.metrics = AgentMetrics(context_settings=self.context_settings)

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
        last_context_level = ContextLevel.NORMAL

        for _ in range(150):
            self.metrics.iterations += 1

            response, metrics = self.llm.complete_stream(
                messages=messages,
                tools=get_tool_schemas(),
                on_text=lambda text: self.on_event(TextDelta(text)),
                max_output_tokens=self.context_settings.output_reserve,
            )

            self.metrics.llm_calls.append(metrics)
            context = self.metrics.context_state
            self.on_event(
                ContextUsage(
                    iteration=self.metrics.iterations,
                    input_tokens=context.input_tokens,
                    context_window=self.metrics.context_window,
                    utilization_percent=context.utilization_percent,
                )
            )

            utilization = context.input_utilization_percent
            if utilization is not None:
                level = context.level
                if (
                    level == ContextLevel.CRITICAL and last_context_level != ContextLevel.CRITICAL
                ) or (level == ContextLevel.WARNING and last_context_level == ContextLevel.NORMAL):
                    self.on_event(
                        ContextWarning(
                            iteration=self.metrics.iterations,
                            level=level,
                            input_tokens=metrics.input_tokens,
                            input_capacity=self.context_settings.input_capacity,
                            utilization_percent=utilization,
                        )
                    )
                last_context_level = level

            if metrics.finish_reason == "length":
                raise RuntimeError("Response incomplete: output token limit reached")

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
