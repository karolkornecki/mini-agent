from pathlib import Path

import pytest

from mini_agent.tools.glob import glob
from mini_agent.tools.grep import grep
from mini_agent.tools.list_files import list_files
from mini_agent.tools.read_file import read_file


def test_list_files(sample_project: Path):
    assert list_files(".") == ["README.md", "src"]


def test_list_files_invalid_directory(project: Path):
    with pytest.raises(ValueError, match="Not a directory"):
        list_files("missing")


def test_read_file(sample_project: Path):
    assert read_file("src/main.py") == ("class Agent:\n    pass\n")


def test_read_file_missing(project: Path):
    with pytest.raises(ValueError, match="Not a file"):
        read_file("missing.py")


def test_read_file_too_large(project: Path):
    (project / "large.txt").write_text("x" * 100_001)

    with pytest.raises(ValueError, match="File too large"):
        read_file("large.txt")


@pytest.mark.parametrize(
    "path",
    [
        ".env",
        ".env.production",
        ".git/config",
        "node_modules/package/index.js",
        "../outside.txt",
    ],
)
def test_blocked_paths(project: Path, path: str):
    with pytest.raises(ValueError):
        read_file(path)


def test_glob(sample_project: Path):
    assert glob("**/*.py") == [
        "src/main.py",
        "src/utils.py",
    ]


def test_grep(sample_project: Path):
    assert grep("class Agent") == ["src/main.py:1:class Agent:"]


def test_grep_regex(sample_project: Path):
    assert grep(r"def \w+", "src") == ["src/utils.py:1:def helper():"]


def test_grep_no_matches(sample_project: Path):
    assert grep("NonExistentClass") == []


def test_symlink_outside_project(
    project: Path,
    tmp_path_factory: pytest.TempPathFactory,
):
    outside = tmp_path_factory.mktemp("outside")
    secret = outside / "secret.txt"
    secret.write_text("secret")

    (project / "link.txt").symlink_to(secret)

    with pytest.raises(ValueError, match="outside project"):
        read_file("link.txt")
