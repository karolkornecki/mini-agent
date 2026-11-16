import os

from dotenv import load_dotenv

from mini_agent.agent.agent import Agent
from mini_agent.cli.cli import run_cli
from mini_agent.cli.renderer import CLIRenderer
from mini_agent.llm.client import LLMClient
from mini_agent.telemetry.observability import flush


def main() -> None:
    load_dotenv()

    renderer = CLIRenderer()
    agent = Agent(LLMClient(), on_event=renderer.render)

    try:
        run_cli(agent, renderer)
    finally:
        context_window = int(os.getenv("MODEL_CONTEXT_WINDOW", "128000"))

        print("\n--- METRICS ---")
        print(agent.metrics.report(context_window))
        flush()


if __name__ == "__main__":
    main()
