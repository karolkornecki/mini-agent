import mini_agent.tools.paths as paths


def list_files(path: str = ".") -> list[str]:
    directory = paths.resolve_path(path)

    if not directory.is_dir():
        raise ValueError("Not a directory")

    results = []

    for item in sorted(directory.iterdir()):
        try:
            paths.resolve_path(str(item.relative_to(paths.ROOT)))
        except ValueError:
            continue

        results.append(item.name)

    return results[: paths.MAX_RESULTS]
