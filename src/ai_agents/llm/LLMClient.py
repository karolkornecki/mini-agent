import os

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam


class LLMClient:
    def __init__(self) -> None:
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
        )
        self.model = "qwen/qwen3-coder"

    def complete(
        self,
        messages: list[ChatCompletionMessageParam],
    ) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )

        return response.choices[0].message.content or ""
