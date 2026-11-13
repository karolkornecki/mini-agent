from openai.types.chat import ChatCompletionToolParam

from mini_agent.tools._paths import resolve_path


def list_files(path: str) -> list[str]:
    directory = resolve_path(path)

    if not directory.is_dir():
        raise ValueError("Not a directory")

    return sorted(item.name for item in directory.iterdir())


LIST_FILES_TOOL: ChatCompletionToolParam = {
    "type": "function",
    "function": {
        "name": "list_files",
        "description": "List files in a project directory",
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
