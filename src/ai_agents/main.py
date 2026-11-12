from dotenv import load_dotenv
from llm.LLMClient import LLMClient


if __name__ == "__main__":
    load_dotenv()

    llm = LLMClient()

    messages = [
        {
            "role": "system",
            "content": "You are a helpful coding assistant.",
        },
        {
            "role": "user",
            "content": "Explain git rebase in one sentence.",
        },
    ]

    response = llm.complete(messages)
    print(response)
