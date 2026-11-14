import json
from typing import Any

from mini_agent.llm.client import LLMClient
from mini_agent.telemetry.observability import (
    agent_trace,
    tool_trace,
)
from mini_agent.telemetry.telemetry import AgentMetrics
from mini_agent.tools.registry import (
    REGISTRY,
    get_tool_schemas,
)


class Agent:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm
        self.metrics = AgentMetrics()

    def run(self, task: str) -> str:
        self.metrics = AgentMetrics()

        with agent_trace() as trace:
            result = self._run(task)
            trace.update(
                metadata={
                    "iterations": self.metrics.iterations,
                    "tool_calls": self.metrics.tool_calls,
                    "tool_errors": self.metrics.tool_errors,
                }
            )
            return result

    def _run(self, task: str) -> str:
        self.metrics = AgentMetrics()

        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "You are a coding assistant. "
                    "Inspect files before editing. "
                    "Never invent file contents. "
                    "Do not read secrets or environment files."
                ),
            },
            {"role": "user", "content": task},
        ]

        for _ in range(150):
            self.metrics.iterations += 1
            response, metrics = self.llm.complete(
                messages=messages,
                tools=get_tool_schemas(),
            )
            print(response.content)
            print("--------------------")

            self.metrics.llm_calls.append(metrics)

            messages.append(response.model_dump(exclude_none=True))

            if not response.tool_calls:
                return response.content or ""

            for call in response.tool_calls:
                self.metrics.tool_calls += 1

                name = call.function.name
                tool = REGISTRY.get(name)

                try:
                    if tool is None:
                        raise ValueError(f"Unknown tool: {name}")

                    arguments = json.loads(call.function.arguments)

                    if not isinstance(arguments, dict):
                        raise ValueError("Arguments must be an object")

                    print(f"\nTool: {name}")
                    print(
                        json.dumps(
                            arguments,
                            indent=2,
                            ensure_ascii=False,
                        )[:2000]
                    )

                    if tool.requires_approval:
                        approval = input("Execute? [y/N]: ").strip().lower()

                        if approval != "y":
                            raise PermissionError("User rejected tool execution")

                    with tool_trace(name) as observation:
                        try:
                            result = tool.execute(**arguments)
                        except Exception as exc:
                            observation.update(
                                level="ERROR",
                                status_message=type(exc).__name__,
                            )
                            raise
                    output = json.dumps(
                        result,
                        ensure_ascii=False,
                    )

                except (
                    ValueError,
                    TypeError,
                    OSError,
                    PermissionError,
                    KeyError,
                ) as exc:
                    self.metrics.tool_errors += 1
                    output = f"Tool error: {exc}"

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": output,
                    }
                )

        raise RuntimeError("Max iterations exceeded")
