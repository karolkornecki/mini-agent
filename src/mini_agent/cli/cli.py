import os
from collections.abc import Callable

from mini_agent.agent.agent import Agent
from mini_agent.cli.renderer import CLIRenderer


def run_cli(
    agent: Agent,
    renderer: CLIRenderer,
    read_input: Callable[[str], str] = input,
) -> None:
    context_window = int(os.getenv("MODEL_CONTEXT_WINDOW", "128000"))

    print("Mini Agent")
    print("Type /exit to quit.\n")

    while True:
        try:
            task = read_input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if task == "/exit":
            print("Goodbye!")
            break

        if task == "/clear":
            agent.reset_conversation()
            print("Conversation cleared.")
            continue

        if not task:
            continue

        try:
            agent.run(task)
        except Exception as exc:
            print(f"\nError: {exc}")
        finally:
            renderer.finish()
            renderer.render_metrics(
                agent.metrics,
                context_window,
            )

        print()
