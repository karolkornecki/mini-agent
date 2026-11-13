from dotenv import load_dotenv

from mini_agent.agent.agent import Agent
from mini_agent.llm.client import LLMClient


def main() -> None:
    load_dotenv()

    agent = Agent(LLMClient())

    result = agent.run(
        "Analyze the architecture of this Python project. "
        "Read relevant source files and explain "
        "how the components interact."
    )

    print("\nAgent response:")
    print(result)


if __name__ == "__main__":
    main()
