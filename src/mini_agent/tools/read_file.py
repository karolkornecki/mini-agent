from openai.types.chat import ChatCompletionToolParam

from mini_agent.tools._paths import resolve_path

MAX_FILE_SIZE = 100_000


def read_file(path: str) -> str:
    file = resolve_path(path)

    if not file.is_file():
        raise ValueError("Not a file")

    if file.stat().st_size > MAX_FILE_SIZE:
        raise ValueError("File too large")

    return file.read_text(encoding="utf-8")


READ_FILE_TOOL: ChatCompletionToolParam = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read a text file from the project",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    },
}
