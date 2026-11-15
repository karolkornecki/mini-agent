from collections.abc import Callable
from typing import Any


def request_approval(
    tool_name: str,
    arguments: dict[str, Any],
    ask: Callable[[str], str] = input,
) -> bool:
    print(f"\nTool: {tool_name}")

    for name, value in arguments.items():
        print(f"  {name}: {str(value)[:200]}")

    answer = ask("Allow execution? [y/N]: ")
    return answer.strip().lower() == "y"
