import subprocess

from mini_agent.tools.paths import ROOT

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
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    return (
        f"Exit code: {result.returncode}\n"
        f"STDOUT:\n{result.stdout[:10000]}\n"
        f"STDERR:\n{result.stderr[:10000]}"
    )
