from pathlib import Path

import pytest

from mini_agent.tools.paths import resolve_path


@pytest.mark.integration
def test_workspace_is_isolated(sample_project: Path) -> None:
    main_file = resolve_path("src/main.py")

    assert main_file == sample_project / "src/main.py"
    assert main_file.exists()
    assert "class Agent:" in main_file.read_text(encoding="utf-8")


@pytest.mark.integration
def test_workspace_blocks_path_traversal(project: Path) -> None:
    with pytest.raises(ValueError, match="Path outside project"):
        resolve_path("../outside.txt")
