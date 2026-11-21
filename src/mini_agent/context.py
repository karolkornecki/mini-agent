from dataclasses import dataclass
from enum import StrEnum


class ContextLevel(StrEnum):
    UNKNOWN = "unknown"
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass(frozen=True)
class ContextSettings:
    context_window: int = 128_000
    output_reserve: int = 8192
    warning_percent: int = 80
    critical_percent: int = 95

    def __post_init__(self) -> None:
        if self.context_window <= 0:
            raise ValueError("context_window must be a positive integer")
        if not 0 < self.output_reserve < self.context_window:
            raise ValueError("output_reserve must be positive and smaller than context_window")
        if not 0 < self.warning_percent < self.critical_percent <= 100:
            raise ValueError(
                "thresholds must satisfy 0 < warning_percent < critical_percent <= 100"
            )

    @property
    def input_capacity(self) -> int:
        return self.context_window - self.output_reserve


@dataclass(frozen=True)
class ContextState:
    settings: ContextSettings
    input_tokens: int | None

    @property
    def utilization_percent(self) -> float | None:
        if self.input_tokens is None:
            return None
        return self.input_tokens * 100 / self.settings.context_window

    @property
    def input_utilization_percent(self) -> float | None:
        if self.input_tokens is None:
            return None
        return self.input_tokens * 100 / self.settings.input_capacity

    @property
    def remaining_input_tokens(self) -> int | None:
        if self.input_tokens is None:
            return None
        return self.settings.input_capacity - self.input_tokens

    @property
    def level(self) -> ContextLevel:
        if self.input_tokens is None:
            return ContextLevel.UNKNOWN
        # Compare counts directly so threshold boundaries do not depend on rounding.
        scaled_input = self.input_tokens * 100
        if scaled_input >= self.settings.critical_percent * self.settings.input_capacity:
            return ContextLevel.CRITICAL
        if scaled_input >= self.settings.warning_percent * self.settings.input_capacity:
            return ContextLevel.WARNING
        return ContextLevel.NORMAL
