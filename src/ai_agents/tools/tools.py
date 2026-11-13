import subprocess
from pathlib import Path

from openai.types.chat import ChatCompletionToolParam

PROJECT_ROOT = Path.cwd().resolve()

MAX_FILE_SIZE = 100_000
MAX_OUTPUT_SIZE = 20_000

ALLOWED_COMMANDS = {
    "git_status": ["git", "status", "--short"],
    "git_diff": ["git", "diff", "--"],
    "pytest": ["uv", "run", "pytest", "-q"],
    "ruff": ["uv", "run", "ruff", "check", "."],
}


BLOCKED_NAMES = {
    ".env",
    ".env.local",
    ".env.production",
    ".git",
    ".ssh",
    ".aws",
    ".venv",
    "__pycache__",
}


def resolve_path(path: str) -> Path:
    target = (PROJECT_ROOT / path).resolve()

    if not target.is_relative_to(PROJECT_ROOT):
        raise ValueError("Path outside project directory")

    relative_parts = target.relative_to(PROJECT_ROOT).parts

    if any(part in BLOCKED_NAMES or part.startswith(".env.") for part in relative_parts):
        raise ValueError("Access denied")

    return target


def list_files(path: str) -> list[str]:
    directory = resolve_path(path)

    if not directory.is_dir():
        raise ValueError("Not a directory")

    return sorted(item.name for item in directory.iterdir())


def read_file(path: str) -> str:
    file = resolve_path(path)

    if not file.is_file():
        raise ValueError("Not a file")

    if file.stat().st_size > MAX_FILE_SIZE:
        raise ValueError("File too large")

    return file.read_text(encoding="utf-8")


def run_command(command: str) -> str:
    if command not in ALLOWED_COMMANDS:
        raise ValueError(f"Command not allowed: {command}")

    result = subprocess.run(
        ALLOWED_COMMANDS[command],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    output = f"Exit code: {result.returncode}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"

    return output[:MAX_OUTPUT_SIZE]


TOOLS: list[ChatCompletionToolParam] = [
    {
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
    },
    {
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
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Run an approved project command",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "enum": list(ALLOWED_COMMANDS),
                    },
                },
                "required": ["command"],
                "additionalProperties": False,
            },
        },
    },
]
