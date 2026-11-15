import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from mini_agent.tools.run_command import run_command


def test_run_command(project: Path):
    completed = subprocess.CompletedProcess(
        args=["git", "status", "--short"],
        returncode=0,
        stdout=" M main.py\n",
        stderr="",
    )

    with patch(
        "mini_agent.tools.run_command.subprocess.run",
        return_value=completed,
    ) as mock_run:
        result = run_command("git_status")

    assert "Exit code: 0" in result
    assert "M main.py" in result

    mock_run.assert_called_once_with(
        ["git", "status", "--short"],
        cwd=project,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def test_command_not_allowed(project: Path):
    with pytest.raises(ValueError, match="not allowed"):
        run_command("rm -rf /")


def test_command_timeout(project: Path):
    with patch(
        "mini_agent.tools.run_command.subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd=["git", "status"], timeout=30),
    ):
        with pytest.raises(subprocess.TimeoutExpired):
            run_command("git_status")
