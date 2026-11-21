from dataclasses import dataclass, field
from time import perf_counter


@dataclass
class LLMCallMetrics:
    latency_seconds: float
    input_tokens: int
    output_tokens: int
    cached_tokens: int = 0
    usage_available: bool = True


@dataclass
class AgentMetrics:
    context_window: int = 128_000
    started_at: float = field(default_factory=perf_counter)
    llm_calls: list[LLMCallMetrics] = field(default_factory=list)
    iterations: int = 0
    tool_calls: int = 0
    tool_errors: int = 0

    def __post_init__(self) -> None:
        if self.context_window <= 0:
            raise ValueError("context_window must be a positive integer")

    @property
    def context_input_tokens(self) -> int | None:
        if not self.llm_calls or not self.llm_calls[-1].usage_available:
            return None
        return self.llm_calls[-1].input_tokens

    @property
    def context_utilization_percent(self) -> float | None:
        input_tokens = self.context_input_tokens
        if input_tokens is None:
            return None
        return input_tokens / self.context_window * 100

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
        utilization = self.context_utilization_percent
        context = f"{utilization:.1f}%" if utilization is not None else "unknown"

        return (
            f"Duration: {self.elapsed_seconds:.2f}s\n"
            f"Iterations: {self.iterations}\n"
            f"LLM calls: {len(self.llm_calls)}\n"
            f"Tool calls: {self.tool_calls}\n"
            f"Tool errors: {self.tool_errors}\n"
            f"Input tokens total: {self.input_tokens_total}\n"
            f"Output tokens total: {self.output_tokens_total}\n"
            f"Cached tokens total: {self.cached_tokens_total}\n"
            f"Context utilization (last request): {context}"
        )
