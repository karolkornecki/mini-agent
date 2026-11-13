from mini_agent.tools.paths import (
    MAX_FILE_SIZE,
    resolve_path,
)


def read_file(path: str) -> str:
    target = resolve_path(path)

    if not target.is_file():
        raise ValueError("Not a file")

    if target.stat().st_size > MAX_FILE_SIZE:
        raise ValueError("File too large")

    return target.read_text(encoding="utf-8")
