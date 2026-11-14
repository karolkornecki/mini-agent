import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from langfuse import get_client

logger = logging.getLogger(__name__)


class NoOpObservation:
    def update(self, **kwargs: Any) -> None:
        pass


@contextmanager
def _observation(
    name: str,
    as_type: str,
    **kwargs: Any,
) -> Iterator[Any]:
    try:
        manager = get_client().start_as_current_observation(
            name=name,
            as_type=as_type,
            **kwargs,
        )
    except Exception:
        logger.exception("Failed to initialize Langfuse observation")
        yield NoOpObservation()
        return

    with manager as observation:
        yield observation


def agent_trace():
    return _observation("agent.run", "agent")


def llm_trace(model: str):
    return _observation(
        "llm.complete",
        "generation",
        model=model,
    )


def tool_trace(name: str):
    return _observation(f"tool.{name}", "tool")


def flush() -> None:
    try:
        get_client().flush()
    except Exception:
        logger.exception("Failed to flush Langfuse")
