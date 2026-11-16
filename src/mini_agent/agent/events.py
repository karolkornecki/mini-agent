from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TextDelta:
    text: str


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


type AgentEvent = TextDelta | ToolStarted | ToolFinished
