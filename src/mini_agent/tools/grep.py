import re

from mini_agent.tools.paths import (
    MAX_FILE_SIZE,
    MAX_RESULTS,
    ROOT,
    resolve_path,
)


def grep(pattern: str, path: str = ".") -> list[str]:
    regex = re.compile(pattern)
    target = resolve_path(path)

    files = [target] if target.is_file() else target.rglob("*")

    matches = []

    for file in files:
        try:
            safe_file = resolve_path(str(file.relative_to(ROOT)))

            if not safe_file.is_file():
                continue

            if safe_file.stat().st_size > MAX_FILE_SIZE:
                continue

            lines = safe_file.read_text(encoding="utf-8").splitlines()

            for number, line in enumerate(lines, start=1):
                if regex.search(line):
                    relative = safe_file.relative_to(ROOT)

                    matches.append(f"{relative}:{number}:{line[:300]}")

                if len(matches) >= MAX_RESULTS:
                    return matches

        except (UnicodeError, OSError, ValueError):
            continue

    return matches
