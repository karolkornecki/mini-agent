
from contextlib import contextmanager
from typing import Iterator

from langfuse import get_client


def agent_trace() -> object:
    return get_client().start_as_current_observation(
        name="agent.run",
        as_type="agent",
    )


def llm_trace(model: str) -> object:
    return get_client().start_as_current_observation(
        name="llm.complete",
        as_type="generation",
        model=model,
    )


def tool_trace(name: str) -> object:
    return get_client().start_as_current_observation(
        name=f"tool.{name}",
        as_type="tool",
    )


def flush() -> None:
    get_client().flush()
