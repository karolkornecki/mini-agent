from pathlib import Path

from openai.types.chat import ChatCompletionToolParam

TOOLS: list[ChatCompletionToolParam] = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files in a project directory",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": ("Directory relative to project root"),
                    }
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    }
]

PROJECT_ROOT = Path.cwd().resolve()


def list_files(path: str) -> list[str]:
    target = (PROJECT_ROOT / path).resolve()

    if not target.is_relative_to(PROJECT_ROOT):
        raise ValueError("Path outside project directory")

    if not target.is_dir():
        raise ValueError("Not a directory")

    return sorted(item.name for item in target.iterdir())
