from openai.types.chat import ChatCompletionToolParam

from mini_agent.tools.list_files import LIST_FILES_TOOL, list_files
from mini_agent.tools.read_file import READ_FILE_TOOL, read_file
from mini_agent.tools.run_command import RUN_COMMAND_TOOL, run_command

__all__ = ["TOOLS", "list_files", "read_file", "run_command"]

TOOLS: list[ChatCompletionToolParam] = [
    LIST_FILES_TOOL,
    READ_FILE_TOOL,
    RUN_COMMAND_TOOL,
]
