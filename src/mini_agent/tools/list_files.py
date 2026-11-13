from mini_agent.tools.paths import (
    MAX_RESULTS,
    ROOT,
    resolve_path,
)


def list_files(path: str = ".") -> list[str]:
    directory = resolve_path(path)

    if not directory.is_dir():
        raise ValueError("Not a directory")

    results = []

    for item in sorted(directory.iterdir()):
        try:
            resolve_path(str(item.relative_to(ROOT)))
        except ValueError:
            continue

        results.append(item.name)

    return results[:MAX_RESULTS]
