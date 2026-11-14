from enum import StrEnum


class Permission(StrEnum):
    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


PERMISSIONS: dict[str, Permission] = {
    "list_files": Permission.ALLOW,
    "read_file": Permission.ALLOW,
    "glob": Permission.ALLOW,
    "grep": Permission.ALLOW,
    "write_file": Permission.ASK,
    "edit_file": Permission.ASK,
    "apply_patch": Permission.ASK,
    "run_command": Permission.ASK,
}


def get_permission(tool_name: str) -> Permission:
    return PERMISSIONS.get(tool_name, Permission.DENY)
