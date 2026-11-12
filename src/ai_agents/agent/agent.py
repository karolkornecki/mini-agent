import json

from openai.types.chat import ChatCompletionMessageParam

from ai_agents.llm.LLMClient import LLMClient
from ai_agents.tools.list_files import TOOLS, list_files


class Agent:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def run(self, task: str) -> str:
        messages: list[ChatCompletionMessageParam] = [
            {
                "role": "system",
                "content": (
                    "You are a coding assistant. "
                    "Use available tools to inspect the project. "
                    "Never invent file contents."
                ),
            },
            {
                "role": "user",
                "content": task,
            },
        ]

        max_iterations = 10

        for _ in range(max_iterations):
            response = self.llm.complete(
                messages=messages,
                tools=TOOLS,
            )

            # Save assistant response, including tool calls
            messages.append(response.model_dump(exclude_none=True))

            # No tool calls means the agent has finished
            if not response.tool_calls:
                return response.content or ""

            for tool_call in response.tool_calls:
                name = tool_call.function.name

                print(f"Executing tool: {name}")

                try:
                    arguments = json.loads(tool_call.function.arguments)

                    if name == "list_files":
                        result = list_files(**arguments)
                    else:
                        raise ValueError(f"Unknown tool: {name}")

                    output = json.dumps(result)

                except (ValueError, TypeError, OSError) as exc:
                    output = f"Tool error: {exc}"

                # Send tool result back to the model
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": output,
                    }
                )

        raise RuntimeError("Agent exceeded maximum iterations")
