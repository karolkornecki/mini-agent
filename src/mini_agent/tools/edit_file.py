from mini_agent.tools import paths
from mini_agent.tools.read_file import read_file


def edit_file(
    path: str,
    old_text: str,
    new_text: str,
) -> str:
    target = paths.resolve_path(path)
    content = read_file(path)

    if not old_text:
        raise ValueError("old_text cannot be empty")

    occurrences = content.count(old_text)

    if occurrences != 1:
        raise ValueError(f"Expected one occurrence, found {occurrences}")

    updated = content.replace(old_text, new_text, 1)

    if len(updated.encode("utf-8")) > paths.MAX_FILE_SIZE:
        raise ValueError("Updated file too large")

    target.write_text(updated, encoding="utf-8")

    return f"Updated {target.relative_to(paths.ROOT)}"
