import os

from dotenv import load_dotenv

from mini_agent.agent.agent import Agent
from mini_agent.llm.client import LLMClient
from mini_agent.telemetry.observability import flush


def main() -> None:
    load_dotenv()

    agent = Agent(LLMClient())

    try:
        result = agent.run(
            # """
            # find bug in calculator add function and fix it,
            # dont add any other function,
            # write tests for it,
            # dont call unnecessary functions,
            # Do not modify any other files.
            # Do not skip any steps.
            # """
            # """
            # analyse this repositroy create directory x and file architecture.txt
            #  with description of
            # all the files and directory structure
            # Do not modify any other files.
            # Do not skip any steps.
            # """
            """
            analyse this repository and do the code review
            Do not modify any other files.
            Do not skip any steps.
            """
        )

        print("\n--- AGENT RESPONSE ---")
        print(result)

    finally:
        context_window = int(os.getenv("MODEL_CONTEXT_WINDOW", "128000"))

        print("\n--- METRICS ---")
        print(agent.metrics.report(context_window))
        flush()


if __name__ == "__main__":
    main()
