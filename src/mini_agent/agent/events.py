from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TextDelta:
    text: str


@dataclass(frozen=True)
class ContextUsage:
    iteration: int
    input_tokens: int | None
    context_window: int
    utilization_percent: float | None


@dataclass(frozen=True)
class ToolStarted:
    call_id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ToolFinished:
    call_id: str
    name: str
    status: str
    output: str


type AgentEvent = TextDelta | ContextUsage | ToolStarted | ToolFinished
