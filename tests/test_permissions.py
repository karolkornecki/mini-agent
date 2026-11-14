import pytest

from mini_agent.permissions.permissions import (
    Permission,
    get_permission,
)


@pytest.mark.parametrize(
    ("tool_name", "expected"),
    [
        ("read_file", Permission.ALLOW),
        ("grep", Permission.ALLOW),
        ("write_file", Permission.ASK),
        ("apply_patch", Permission.ASK),
        ("run_command", Permission.ASK),
        ("unknown_tool", Permission.DENY),
    ],
)
def test_permissions(tool_name, expected):
    assert get_permission(tool_name) == expected
