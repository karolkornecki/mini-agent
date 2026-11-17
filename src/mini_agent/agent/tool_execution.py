import json
from collections.abc import Callable
from subprocess import TimeoutExpired
from typing import Any

from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageFunctionToolCall,
)

from mini_agent.agent.events import AgentEvent, ToolFinished, ToolStarted
from mini_agent.permissions.permissions import Permission, get_permission
from mini_agent.telemetry.observability import tool_trace
from mini_agent.telemetry.telemetry import AgentMetrics
from mini_agent.tools.registry import REGISTRY


class ApprovalRejected(Exception):
    """The user declined a tool call that requires approval."""


def execute_tool(
    call: ChatCompletionMessageFunctionToolCall,
    *,
    metrics: AgentMetrics,
    on_event: Callable[[AgentEvent], None],
    approve: Callable[[str, dict[str, Any]], bool],
) -> dict[str, Any]:
    """Execute one call, record its outcome, and return its complete tool-result message."""
    metrics.tool_calls += 1
    name = call.function.name
    arguments: dict[str, Any] = {}
    parse_error: str | None = None

    try:
        parsed = json.loads(call.function.arguments)
        if not isinstance(parsed, dict):
            raise ValueError("Arguments must be an object")
        arguments = parsed
    except (ValueError, TypeError) as exc:
        parse_error = f"Tool error: {exc}"

    on_event(ToolStarted(call_id=call.id, name=name, arguments=arguments))

    status = "success"
    if parse_error is not None:
        status = "error"
        output = parse_error
    else:
        try:
            tool = REGISTRY.get(name)
            if tool is None:
                raise ValueError(f"Unknown tool: {name}")

            result = execute_with_permission(
                tool.execute,
                name=name,
                arguments=arguments,
                approve=approve,
            )
            output = json.dumps(result, ensure_ascii=False)
        except ApprovalRejected as exc:
            status = "denied"
            output = str(exc)
        except (ValueError, TypeError, OSError, KeyError, TimeoutExpired) as exc:
            status = "error"
            output = f"Tool error: {exc}"

    if status == "error":
        metrics.tool_errors += 1

    on_event(ToolFinished(call_id=call.id, name=name, status=status, output=output))
    return {"role": "tool", "tool_call_id": call.id, "content": output}


def execute_with_permission(
    execute: Callable[..., Any],
    *,
    name: str,
    arguments: dict[str, Any],
    approve: Callable[[str, dict[str, Any]], bool],
) -> Any:
    """Check permission and approval before tracing and invoking the callable."""
    permission = get_permission(name)
    if permission == Permission.DENY:
        raise PermissionError(f"Tool denied: {name}")

    if permission == Permission.ASK and not approve(name, arguments):
        raise ApprovalRejected(f"User rejected execution of {name}")

    with tool_trace(name) as observation:
        try:
            return execute(**arguments)
        except Exception as exc:
            observation.update(
                level="ERROR",
                status_message=type(exc).__name__,
            )
            raise
