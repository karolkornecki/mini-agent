from mini_agent.tools.registry import (
    REGISTRY,
    get_tool_schemas,
)

EXPECTED_TOOLS = {
    "list_files",
    "read_file",
    "glob",
    "grep",
    "write_file",
    "edit_file",
    "apply_patch",
    "run_command",
}


def test_all_tools_registered():
    assert set(REGISTRY) == EXPECTED_TOOLS


def test_unique_tool_names():
    schemas = get_tool_schemas()
    names = [schema["function"]["name"] for schema in schemas]

    assert len(names) == len(set(names))


def test_all_tools_have_valid_schema():
    for schema in get_tool_schemas():
        assert schema["type"] == "function"

        function = schema["function"]
        parameters = function["parameters"]

        assert function["name"]
        assert function["description"]
        assert parameters["type"] == "object"
        assert parameters["additionalProperties"] is False

        assert set(parameters["required"]).issubset(parameters["properties"])
