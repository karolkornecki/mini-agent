from typing import Any


def remove_oldest_turn(messages: list[dict[str, Any]]) -> bool:
    """Drop the oldest turn after the active user message has been appended."""
    oldest_start = None
    for index, message in enumerate(messages):
        if message["role"] != "user":
            continue
        if oldest_start is not None:
            del messages[oldest_start:index]
            return True
        oldest_start = index
    return False
