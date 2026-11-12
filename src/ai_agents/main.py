from dotenv import load_dotenv

from ai_agents.agent.agent import Agent
from ai_agents.llm.LLMClient import LLMClient

if __name__ == "__main__":
    load_dotenv()

    llm = LLMClient()
    agent = Agent(llm)

    result = agent.run("List files and explain the project structure.")

    print("\nAgent response:")
    print(result)
