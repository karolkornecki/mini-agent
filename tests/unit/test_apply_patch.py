from pathlib import Path

import pytest

from mini_agent.tools.apply_patch import apply_patch


def test_apply_patch(sample_project: Path):
    patch = (
        "--- a/src/main.py\n"
        "+++ b/src/main.py\n"
        "@@ -1,2 +1,2 @@\n"
        " class Agent:\n"
        "-    pass\n"
        "+    return 42\n"
    )

    result = apply_patch(patch)

    assert result == "Patched src/main.py"
    assert (sample_project / "src/main.py").read_text() == "class Agent:\n    return 42\n"


def test_apply_patch_context_mismatch(sample_project: Path):
    original = (sample_project / "src/main.py").read_text()

    patch = (
        "--- a/src/main.py\n"
        "+++ b/src/main.py\n"
        "@@ -1,2 +1,2 @@\n"
        " class WrongName:\n"
        "-    pass\n"
        "+    return 42\n"
    )

    with pytest.raises(ValueError, match="context mismatch"):
        apply_patch(patch)

    assert (sample_project / "src/main.py").read_text() == original


def test_apply_patch_rename_rejected(sample_project: Path):
    source = sample_project / "src/main.py"
    destination = sample_project / "src/renamed.py"
    original = source.read_bytes()

    patch = (
        "--- a/src/main.py\n"
        "+++ b/src/renamed.py\n"
        "@@ -1,2 +1,2 @@\n"
        " class Agent:\n"
        "-    pass\n"
        "+    value = 42\n"
    )

    with pytest.raises(ValueError, match="Renaming unsupported"):
        apply_patch(patch)

    assert source.read_bytes() == original
    assert not destination.exists()


def test_apply_patch_multiple_files(sample_project: Path):
    patch = (
        "--- a/src/main.py\n"
        "+++ b/src/main.py\n"
        "@@ -1 +1 @@\n"
        "-class Agent:\n"
        "+class NewAgent:\n"
        "--- a/src/utils.py\n"
        "+++ b/src/utils.py\n"
        "@@ -1 +1 @@\n"
        "-def helper():\n"
        "+def other():\n"
    )

    with pytest.raises(ValueError, match="exactly one file"):
        apply_patch(patch)
