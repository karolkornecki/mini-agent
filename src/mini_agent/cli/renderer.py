import json

from mini_agent.agent.events import (
    AgentEvent,
    ContextUsage,
    ContextWarning,
    TextDelta,
    ToolFinished,
    ToolStarted,
)
from mini_agent.telemetry.telemetry import AgentMetrics


class CLIRenderer:
    def __init__(self) -> None:
        self._streaming = False

    def render(self, event: AgentEvent) -> None:
        match event:
            case TextDelta(text=text):
                print(text, end="", flush=True)
                self._streaming = True

            case ContextUsage(
                iteration=iteration,
                input_tokens=input_tokens,
                context_window=context_window,
                utilization_percent=utilization,
            ):
                self._finish_stream()
                context = "unknown"
                if input_tokens is not None and utilization is not None:
                    context = f"{input_tokens:,} / {context_window:,} tokens ({utilization:.1f}%)"
                print(f"Context (last request, iteration {iteration}): {context}")

            case ContextWarning(
                iteration=iteration,
                level=level,
                input_tokens=input_tokens,
                input_capacity=input_capacity,
                utilization_percent=utilization,
            ):
                self._finish_stream()
                print(
                    f"! Context {level} (iteration {iteration}): "
                    f"{input_tokens:,} / {input_capacity:,} input tokens "
                    f"({utilization:.1f}% of input capacity)"
                )

            case ToolStarted(name=name, arguments=arguments):
                self._finish_stream()
                print(f"\n→ {name}")
                print(
                    json.dumps(
                        arguments,
                        indent=2,
                        ensure_ascii=False,
                    )
                )

            case ToolFinished(name=name, status=status, output=output):
                marker = {
                    "success": "✓",
                    "error": "✗",
                    "denied": "!",
                }.get(status, "?")

                print(f"{marker} {name} [{status}]")

                if status != "success":
                    print(f"  {output}")

    def finish(self) -> None:
        self._finish_stream()

    def _finish_stream(self) -> None:
        if self._streaming:
            print()
            self._streaming = False

    def render_metrics(
        self,
        metrics: AgentMetrics,
    ) -> None:
        self.finish()
        print("\n── Task metrics ───────────────────")
        print(metrics.report())
        print("───────────────────────────────────")
