import os

from dotenv import load_dotenv

from mini_agent.agent.agent import Agent
from mini_agent.cli.cli import run_cli
from mini_agent.cli.renderer import CLIRenderer
from mini_agent.context import ContextSettings
from mini_agent.llm.client import LLMClient
from mini_agent.telemetry.observability import flush


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        raise ValueError(f"{name} must be an integer") from None


def main() -> None:
    load_dotenv()
    defaults = ContextSettings()
    try:
        context_settings = ContextSettings(
            context_window=_env_int("MODEL_CONTEXT_WINDOW", defaults.context_window),
            output_reserve=_env_int("MODEL_OUTPUT_RESERVE", defaults.output_reserve),
            warning_percent=_env_int("CONTEXT_WARNING_PERCENT", defaults.warning_percent),
            critical_percent=_env_int("CONTEXT_CRITICAL_PERCENT", defaults.critical_percent),
            max_tool_output_chars=_env_int("MAX_TOOL_OUTPUT_CHARS", defaults.max_tool_output_chars),
        )
    except ValueError as exc:
        raise SystemExit(f"Invalid context configuration: {exc}") from None

    renderer = CLIRenderer()
    agent = Agent(
        LLMClient(),
        on_event=renderer.render,
        context_settings=context_settings,
    )

    try:
        run_cli(agent, renderer)
    finally:
        flush()


if __name__ == "__main__":
    main()
