import mini_agent.tools.paths as paths


def write_file(path: str, content: str) -> str:
    target = paths.resolve_path(path)

    if target.exists():
        raise ValueError("File already exists")

    if len(content.encode("utf-8")) > paths.MAX_FILE_SIZE:
        raise ValueError("Content too large")

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")

    return f"Created {target.relative_to(paths.ROOT)}"
