from pathlib import Path

import pytest

from mini_agent.tools.edit_file import edit_file
from mini_agent.tools.write_file import write_file


def test_write_file(project: Path):
    result = write_file("src/new.py", "x = 42\n")

    assert result == "Created src/new.py"
    assert (project / "src/new.py").read_text() == "x = 42\n"


def test_write_file_does_not_overwrite(project: Path):
    target = project / "existing.py"
    target.write_text("original")

    with pytest.raises(ValueError, match="already exists"):
        write_file("existing.py", "replacement")

    assert target.read_text() == "original"


def test_write_file_too_large(project: Path):
    with pytest.raises(ValueError, match="Content too large"):
        write_file("large.py", "x" * 100_001)


def test_edit_file(sample_project: Path):
    result = edit_file(
        "src/main.py",
        "pass",
        "return 42",
    )

    assert result == "Updated src/main.py"
    assert "return 42" in (sample_project / "src/main.py").read_text()


def test_edit_file_no_match(sample_project: Path):
    with pytest.raises(ValueError, match="Expected one occurrence, found 0"):
        edit_file("src/main.py", "missing", "replacement")


def test_edit_file_multiple_matches(project: Path):
    target = project / "test.py"
    target.write_text("foo\nfoo\n")

    with pytest.raises(ValueError, match="Expected one occurrence, found 2"):
        edit_file("test.py", "foo", "bar")

    assert target.read_text() == "foo\nfoo\n"


def test_edit_file_empty_old_text(sample_project: Path):
    with pytest.raises(ValueError, match="cannot be empty"):
        edit_file("src/main.py", "", "replacement")
