from dataclasses import dataclass, field
from time import perf_counter

from mini_agent.context import ContextSettings, ContextState


@dataclass
class LLMCallMetrics:
    latency_seconds: float
    input_tokens: int
    output_tokens: int
    cached_tokens: int = 0
    usage_available: bool = True
    finish_reason: str | None = None


@dataclass
class AgentMetrics:
    context_settings: ContextSettings = field(default_factory=ContextSettings)
    started_at: float = field(default_factory=perf_counter)
    llm_calls: list[LLMCallMetrics] = field(default_factory=list)
    iterations: int = 0
    tool_calls: int = 0
    tool_errors: int = 0

    @property
    def context_window(self) -> int:
        return self.context_settings.context_window

    @property
    def context_state(self) -> ContextState:
        input_tokens = None
        if self.llm_calls and self.llm_calls[-1].usage_available:
            input_tokens = self.llm_calls[-1].input_tokens
        return ContextState(self.context_settings, input_tokens)

    @property
    def context_input_tokens(self) -> int | None:
        return self.context_state.input_tokens

    @property
    def context_utilization_percent(self) -> float | None:
        return self.context_state.utilization_percent

    @property
    def input_tokens_total(self) -> int:
        return sum(call.input_tokens for call in self.llm_calls)

    @property
    def output_tokens_total(self) -> int:
        return sum(call.output_tokens for call in self.llm_calls)

    @property
    def cached_tokens_total(self) -> int:
        return sum(call.cached_tokens for call in self.llm_calls)

    @property
    def elapsed_seconds(self) -> float:
        return perf_counter() - self.started_at

    def report(self) -> str:
        state = self.context_state
        utilization = state.utilization_percent
        context = f"{utilization:.1f}%" if utilization is not None else "unknown"
        input_utilization = state.input_utilization_percent
        input_context = f"{input_utilization:.1f}%" if input_utilization is not None else "unknown"
        remaining = state.remaining_input_tokens
        remaining_context = f"{remaining:,} tokens" if remaining is not None else "unknown"

        return (
            f"Duration: {self.elapsed_seconds:.2f}s\n"
            f"Iterations: {self.iterations}\n"
            f"LLM calls: {len(self.llm_calls)}\n"
            f"Tool calls: {self.tool_calls}\n"
            f"Tool errors: {self.tool_errors}\n"
            f"Input tokens total: {self.input_tokens_total}\n"
            f"Output tokens total: {self.output_tokens_total}\n"
            f"Cached tokens total: {self.cached_tokens_total}\n"
            f"Context utilization (last request): {context}\n"
            f"Input capacity: {self.context_settings.input_capacity:,} tokens "
            f"({self.context_settings.output_reserve:,} reserved for output)\n"
            f"Input capacity utilization (last request): {input_context}\n"
            f"Remaining input capacity (last request): {remaining_context}\n"
            f"Context level (last request): {state.level}"
        )
