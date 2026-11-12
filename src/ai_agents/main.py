import json

from dotenv import load_dotenv
from llm.LLMClient import LLMClient
from tools.list_files import TOOLS, list_files

if __name__ == "__main__":
    load_dotenv()

    llm = LLMClient()

    messages = [
        {
            "role": "system",
            "content": ("You are a coding assistant. Use tools when needed."),
        },
        {
            "role": "user",
            "content": "List files in tools directory",
        },
    ]

    response = llm.complete(messages, tools=TOOLS)

    if response.tool_calls:
        for tool_call in response.tool_calls:
            name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)

            print(f"Tool requested: {name}")
            print(f"Arguments: {arguments}")

            if name == "list_files":
                result = list_files(**arguments)
                print(f"Tool result: {result}")
            else:
                raise ValueError(f"Unknown tool: {name}")
    else:
        print(response.content)
