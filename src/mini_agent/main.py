import os

from dotenv import load_dotenv

from mini_agent.agent.agent import Agent
from mini_agent.cli.cli import run_cli
from mini_agent.cli.renderer import CLIRenderer
from mini_agent.llm.client import LLMClient
from mini_agent.telemetry.observability import flush


def main() -> None:
    load_dotenv()

    try:
        context_window = int(os.getenv("MODEL_CONTEXT_WINDOW", "128000"))
        if context_window <= 0:
            raise ValueError
    except ValueError:
        raise SystemExit("MODEL_CONTEXT_WINDOW must be a positive integer") from None

    renderer = CLIRenderer()
    agent = Agent(
        LLMClient(),
        on_event=renderer.render,
        context_window=context_window,
    )

    try:
        run_cli(agent, renderer)
    finally:
        flush()


if __name__ == "__main__":
    main()
