
from pathlib import Path

import pytest

from mini_agent.tools import paths


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(paths, "ROOT", tmp_path)
    return tmp_path


@pytest.fixture
def sample_project(project: Path) -> Path:
    src = project / "src"
    src.mkdir()

    (src / "main.py").write_text(
        "class Agent:\n    pass\n",
        encoding="utf-8",
    )
    (src / "utils.py").write_text(
        "def helper():\n    return 42\n",
        encoding="utf-8",
    )
    (project / "README.md").write_text(
        "# Mini Agent\n",
        encoding="utf-8",
    )

    return project
