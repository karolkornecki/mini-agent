
from pathlib import Path

ROOT = Path.cwd().resolve()

MAX_FILE_SIZE = 100_000
MAX_RESULTS = 100

BLOCKED = {
    ".git",
    ".venv",
    ".ssh",
    ".aws",
    "__pycache__",
    "node_modules",
}


def resolve_path(path: str) -> Path:
    target = (ROOT / path).resolve()

    if not target.is_relative_to(ROOT):
        raise ValueError("Path outside project")

    parts = target.relative_to(ROOT).parts

    if any(
        part in BLOCKED
        or part == ".env"
        or part.startswith(".env.")
        for part in parts
    ):
        raise ValueError("Access denied")

    return target
