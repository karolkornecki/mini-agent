from pathlib import Path

PROJECT_ROOT = Path.cwd().resolve()

BLOCKED_NAMES = {
    ".env",
    ".env.local",
    ".env.production",
    ".git",
    ".ssh",
    ".aws",
    ".venv",
    "__pycache__",
}


def resolve_path(path: str) -> Path:
    target = (PROJECT_ROOT / path).resolve()

    if not target.is_relative_to(PROJECT_ROOT):
        raise ValueError("Path outside project directory")

    relative_parts = target.relative_to(PROJECT_ROOT).parts

    if any(part in BLOCKED_NAMES or part.startswith(".env.") for part in relative_parts):
        raise ValueError("Access denied")

    return target
