import json
import subprocess
from collections.abc import Callable
from typing import Any

from openai.types.chat import ChatCompletionMessageParam

from ai_agents.llm.LLMClient import LLMClient
from ai_agents.tools.tools import (
    TOOLS,
    list_files,
    read_file,
    run_command,
)

TOOL_REGISTRY: dict[str, Callable[..., Any]] = {
    "list_files": list_files,
    "read_file": read_file,
    "run_command": run_command,
}


class Agent:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def run(self, task: str) -> str:
        messages: list[ChatCompletionMessageParam] = [
            {
                "role": "system",
                "content": (
                    "You are a coding assistant. "
                    "Inspect the project using available tools. "
                    "Never invent file contents. "
                    "Do not read secrets or environment files."
                ),
            },
            {"role": "user", "content": task},
        ]

        for _ in range(10):
            response = self.llm.complete(
                messages=messages,
                tools=TOOLS,
            )

            messages.append(response.model_dump(exclude_none=True))

            if not response.tool_calls:
                return response.content or ""

            for tool_call in response.tool_calls:
                name = tool_call.function.name
                print(f"Executing: {name}")

                try:
                    arguments = json.loads(tool_call.function.arguments)

                    tool = TOOL_REGISTRY.get(name)
                    if tool is None:
                        raise ValueError(f"Unknown tool: {name}")

                    result = tool(**arguments)
                    output = json.dumps(result)

                except (
                    ValueError,
                    TypeError,
                    OSError,
                    subprocess.TimeoutExpired,
                ) as exc:
                    output = f"Tool error: {exc}"

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": output,
                    }
                )

        raise RuntimeError("Maximum iterations exceeded")
