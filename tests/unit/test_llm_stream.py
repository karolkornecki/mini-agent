from types import SimpleNamespace
from unittest.mock import Mock, patch

from mini_agent.llm.client import LLMClient


def chunk(
    *,
    content=None,
    tool_calls=None,
    usage=None,
):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(
                    content=content,
                    tool_calls=tool_calls,
                )
            )
        ]
        if usage is None
        else [],
        usage=usage,
    )


def tool_delta(index, *, id=None, name=None, arguments=None):
    return SimpleNamespace(
        index=index,
        id=id,
        function=SimpleNamespace(
            name=name,
            arguments=arguments,
        ),
    )


def make_client(chunks):
    llm = LLMClient.__new__(LLMClient)
    llm.model = "test-model"
    llm.client = Mock()
    llm.client.chat.completions.create.return_value = iter(chunks)
    return llm


def test_stream_reconstructs_text():
    llm = make_client(
        [
            chunk(content="Hello"),
            chunk(content=" "),
            chunk(content="world"),
            chunk(
                usage=SimpleNamespace(
                    prompt_tokens=10,
                    completion_tokens=3,
                    prompt_tokens_details=SimpleNamespace(
                        cached_tokens=2,
                    ),
                )
            ),
        ]
    )

    received = []

    with patch("mini_agent.llm.client.llm_trace"):
        response, metrics = llm.complete_stream(
            messages=[{"role": "user", "content": "Hi"}],
            tools=[],
            on_text=received.append,
        )

    assert received == ["Hello", " ", "world"]
    assert response.content == "Hello world"
    assert response.tool_calls is None

    assert metrics.input_tokens == 10
    assert metrics.output_tokens == 3
    assert metrics.cached_tokens == 2
    assert metrics.usage_available is True

    llm.client.chat.completions.create.assert_called_once()
    assert llm.client.chat.completions.create.call_args.kwargs["stream"] is True


def test_stream_reconstructs_tool_calls():
    llm = make_client(
        [
            chunk(
                tool_calls=[
                    tool_delta(
                        0,
                        id="call_123",
                        name="read_file",
                        arguments='{"path":',
                    ),
                ]
            ),
            chunk(
                tool_calls=[
                    tool_delta(
                        0,
                        arguments='"README.md"}',
                    ),
                ]
            ),
        ]
    )

    received = []

    with patch("mini_agent.llm.client.llm_trace"):
        response, metrics = llm.complete_stream(
            messages=[{"role": "user", "content": "Read README"}],
            tools=[],
            on_text=received.append,
        )

    assert received == []
    assert response.content is None
    assert len(response.tool_calls) == 1

    call = response.tool_calls[0]

    assert call.id == "call_123"
    assert call.function.name == "read_file"
    assert call.function.arguments == '{"path":"README.md"}'

    assert metrics.input_tokens == 0
    assert metrics.output_tokens == 0
    assert metrics.cached_tokens == 0
    assert metrics.usage_available is False


def test_reported_zero_usage_is_available() -> None:
    llm = make_client(
        [
            chunk(
                usage=SimpleNamespace(
                    prompt_tokens=0,
                    completion_tokens=0,
                    prompt_tokens_details=None,
                )
            ),
        ]
    )

    with patch("mini_agent.llm.client.llm_trace"):
        _, metrics = llm.complete_stream(messages=[], tools=[], on_text=lambda text: None)

    assert metrics.input_tokens == 0
    assert metrics.output_tokens == 0
    assert metrics.usage_available is True
