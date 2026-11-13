from mini_agent.tools.paths import (
    MAX_RESULTS,
    ROOT,
    resolve_path,
)


def glob(pattern: str) -> list[str]:
    results = []

    for item in ROOT.glob(pattern):
        try:
            target = resolve_path(str(item.relative_to(ROOT)))
        except ValueError:
            continue

        if target.is_file():
            results.append(str(target.relative_to(ROOT)))

        if len(results) >= MAX_RESULTS:
            break

    return sorted(results)
