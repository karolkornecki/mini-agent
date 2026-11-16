import json

from mini_agent.agent.events import (
    AgentEvent,
    TextDelta,
    ToolFinished,
    ToolStarted,
)


class CLIRenderer:
    def __init__(self) -> None:
        self._streaming = False

    def render(self, event: AgentEvent) -> None:
        match event:
            case TextDelta(text=text):
                print(text, end="", flush=True)
                self._streaming = True

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
