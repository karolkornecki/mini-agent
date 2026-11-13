import subprocess

from openai.types.chat import ChatCompletionToolParam

from mini_agent.tools._paths import PROJECT_ROOT

MAX_OUTPUT_SIZE = 20_000

ALLOWED_COMMANDS = {
    "git_status": ["git", "status", "--short"],
    "git_diff": ["git", "diff", "--"],
    "pytest": ["uv", "run", "pytest", "-q"],
    "ruff": ["uv", "run", "ruff", "check", "."],
}


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


RUN_COMMAND_TOOL: ChatCompletionToolParam = {
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
}
