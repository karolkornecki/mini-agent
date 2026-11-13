from typing import Any

from mini_agent.tools.apply_patch import apply_patch
from mini_agent.tools.base import Tool
from mini_agent.tools.edit_file import edit_file
from mini_agent.tools.glob import glob
from mini_agent.tools.grep import grep
from mini_agent.tools.list_files import list_files
from mini_agent.tools.read_file import read_file
from mini_agent.tools.run_command import (
    ALLOWED_COMMANDS,
    run_command,
)
from mini_agent.tools.write_file import write_file


def params(
    properties: dict[str, Any],
    required: list[str],
) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


STRING = {"type": "string"}

TOOLS = [
    Tool(
        "list_files",
        "List files in a directory",
        params({"path": STRING}, []),
        list_files,
    ),
    Tool(
        "read_file",
        "Read a UTF-8 text file",
        params({"path": STRING}, ["path"]),
        read_file,
    ),
    Tool(
        "glob",
        "Find files by glob pattern",
        params({"pattern": STRING}, ["pattern"]),
        glob,
    ),
    Tool(
        "grep",
        "Search files with regex",
        params(
            {"pattern": STRING, "path": STRING},
            ["pattern"],
        ),
        grep,
    ),
    Tool(
        "write_file",
        "Create a new file without overwriting",
        params(
            {"path": STRING, "content": STRING},
            ["path", "content"],
        ),
        write_file,
        requires_approval=True,
    ),
    Tool(
        "edit_file",
        "Replace exactly one text occurrence",
        params(
            {
                "path": STRING,
                "old_text": STRING,
                "new_text": STRING,
            },
            ["path", "old_text", "new_text"],
        ),
        edit_file,
        requires_approval=True,
    ),
    Tool(
        "apply_patch",
        "Apply a unified diff to one existing file",
        params({"patch": STRING}, ["patch"]),
        apply_patch,
        requires_approval=True,
    ),
    Tool(
        "run_command",
        "Execute an allowlisted project command",
        params(
            {
                "command": {
                    "type": "string",
                    "enum": list(ALLOWED_COMMANDS),
                },
            },
            ["command"],
        ),
        run_command,
        requires_approval=True,
    ),
]

REGISTRY = {tool.name: tool for tool in TOOLS}


def get_tool_schemas() -> list[dict[str, Any]]:
    return [tool.schema() for tool in TOOLS]
