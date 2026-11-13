from mini_agent.tools import paths


def glob(pattern: str) -> list[str]:
    results = []

    for item in paths.ROOT.glob(pattern):
        try:
            target = paths.resolve_path(str(item.relative_to(paths.ROOT)))
        except ValueError:
            continue

        if target.is_file():
            results.append(str(target.relative_to(paths.ROOT)))

        if len(results) >= paths.MAX_RESULTS:
            break

    return sorted(results)
