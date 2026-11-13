import mini_agent.tools.paths as paths


def read_file(path: str) -> str:
    target = paths.resolve_path(path)

    if not target.is_file():
        raise ValueError("Not a file")

    if target.stat().st_size > paths.MAX_FILE_SIZE:
        raise ValueError("File too large")

    return target.read_text(encoding="utf-8")
