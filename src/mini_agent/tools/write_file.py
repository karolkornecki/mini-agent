from mini_agent.tools.paths import (
    MAX_FILE_SIZE,
    ROOT,
    resolve_path,
)


def write_file(path: str, content: str) -> str:
    target = resolve_path(path)

    if target.exists():
        raise ValueError("File already exists")

    if len(content.encode("utf-8")) > MAX_FILE_SIZE:
        raise ValueError("Content too large")

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")

    return f"Created {target.relative_to(ROOT)}"
