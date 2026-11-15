from collections.abc import Callable

from mini_agent.agent.agent import Agent


def run_cli(
    agent: Agent,
    read_input: Callable[[str], str] = input,
) -> None:
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
            result = agent.run(task)
            print(f"\n{result}\n")
        except Exception as exc:
            print(f"\nError: {exc}\n")
