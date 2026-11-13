from io import StringIO

from unidiff import PatchSet

from mini_agent.tools.paths import (
    MAX_FILE_SIZE,
    ROOT,
    resolve_path,
)
from mini_agent.tools.read_file import read_file


def apply_patch(patch: str) -> str:
    parsed = PatchSet(StringIO(patch))

    if len(parsed) != 1:
        raise ValueError("Expected exactly one file")

    file_patch = parsed[0]

    if file_patch.is_added_file or file_patch.is_removed_file:
        raise ValueError("File creation/deletion unsupported")

    if file_patch.source_file != file_patch.target_file:
        raise ValueError("Renaming unsupported")

    path = file_patch.path.removeprefix("b/")
    target = resolve_path(path)

    original = read_file(path).splitlines(keepends=True)
    result = []
    cursor = 0

    for hunk in file_patch:
        start = hunk.source_start - 1

        if start < cursor or start > len(original):
            raise ValueError("Invalid hunk position")

        result.extend(original[cursor:start])
        cursor = start

        for line in hunk:
            if line.is_context or line.is_removed:
                if cursor >= len(original):
                    raise ValueError("Patch exceeds file")

                actual = original[cursor].rstrip("\r\n")
                expected = line.value.rstrip("\r\n")

                if actual != expected:
                    raise ValueError("Patch context mismatch")

                if line.is_context:
                    result.append(original[cursor])

                cursor += 1

            elif line.is_added:
                result.append(line.value)

    result.extend(original[cursor:])
    updated = "".join(result)

    if len(updated.encode("utf-8")) > MAX_FILE_SIZE:
        raise ValueError("Updated file too large")

    target.write_text(updated, encoding="utf-8")

    return f"Patched {target.relative_to(ROOT)}"
