from dotenv import load_dotenv

from ai_agents.agent.agent import Agent
from ai_agents.llm.LLMClient import LLMClient

if __name__ == "__main__":
    load_dotenv()

    agent = Agent(LLMClient())

    result = agent.run(
        "Analyze the architecture of this Python project. "
        "Read relevant source files and explain "
        "how the components interact."
    )

    print("\nAgent response:")
    print(result)
