import os
from openai import OpenAI

from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )

    response = client.chat.completions.create(
        model="qwen/qwen3-coder",
        messages=[
            {
                "role": "system",
                "content": "You are a coding assistant."
            },
            {
                "role": "user",
                "content": "Explain what git status does."
            }
        ]
    )

    print(response.choices[0].message.content)
