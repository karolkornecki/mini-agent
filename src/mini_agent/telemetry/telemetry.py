from dataclasses import dataclass, field
from time import perf_counter


@dataclass
class LLMCallMetrics:
    latency_seconds: float
    input_tokens: int
    output_tokens: int
    cached_tokens: int = 0


@dataclass
class AgentMetrics:
    started_at: float = field(default_factory=perf_counter)
    llm_calls: list[LLMCallMetrics] = field(default_factory=list)
    iterations: int = 0
    tool_calls: int = 0
    tool_errors: int = 0

    @property
    def input_tokens_total(self) -> int:
        return sum(x.input_tokens for x in self.llm_calls)

    @property
    def output_tokens_total(self) -> int:
        return sum(x.output_tokens for x in self.llm_calls)

    @property
    def elapsed_seconds(self) -> float:
        return perf_counter() - self.started_at

    def report(self, context_window: int) -> str:
        latest = self.llm_calls[-1] if self.llm_calls else None

        utilization = (
            latest.input_tokens / context_window * 100 if latest and context_window > 0 else 0.0
        )

        return (
            f"Duration: {self.elapsed_seconds:.2f}s\n"
            f"Iterations: {self.iterations}\n"
            f"LLM calls: {len(self.llm_calls)}\n"
            f"Tool calls: {self.tool_calls}\n"
            f"Tool errors: {self.tool_errors}\n"
            f"Input tokens total: {self.input_tokens_total}\n"
            f"Output tokens total: {self.output_tokens_total}\n"
            f"Context utilization: {utilization:.1f}%"
        )
