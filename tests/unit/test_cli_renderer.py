from unittest.mock import Mock

from mini_agent.cli.cli import run_cli
from mini_agent.cli.renderer import CLIRenderer
from mini_agent.telemetry.telemetry import (
    AgentMetrics,
    LLMCallMetrics,
)


def make_agent() -> Mock:
    agent = Mock()
    agent.metrics = AgentMetrics()
    agent.run.return_value = "Done"
    return agent


def test_cli_runs_multiple_tasks() -> None:
    agent = make_agent()
    renderer = Mock()

    commands = iter(
        [
            "read README.md",
            "list files",
            "/exit",
        ]
    )

    run_cli(
        agent,
        renderer,
        read_input=lambda _: next(commands),
    )

    assert agent.run.call_count == 2
    agent.run.assert_any_call("read README.md")
    agent.run.assert_any_call("list files")
    assert renderer.render_metrics.call_count == 2
    assert renderer.finish.call_count == 2


def test_cli_ignores_empty_input() -> None:
    agent = make_agent()
    renderer = Mock()

    commands = iter(["", "   ", "/exit"])

    run_cli(
        agent,
        renderer,
        read_input=lambda _: next(commands),
    )

    agent.run.assert_not_called()
    renderer.render_metrics.assert_not_called()


def test_cli_clears_conversation() -> None:
    agent = make_agent()
    renderer = Mock()

    commands = iter(
        [
            "read README.md",
            "/clear",
            "list files",
            "/exit",
        ]
    )

    run_cli(
        agent,
        renderer,
        read_input=lambda _: next(commands),
    )

    agent.reset_conversation.assert_called_once_with()
    assert agent.run.call_count == 2
    assert renderer.render_metrics.call_count == 2


def test_cli_displays_metrics_after_failure(capsys) -> None:
    agent = make_agent()
    renderer = Mock()

    agent.run.side_effect = RuntimeError("LLM unavailable")

    commands = iter(["analyze repo", "/exit"])

    run_cli(
        agent,
        renderer,
        read_input=lambda _: next(commands),
    )

    output = capsys.readouterr().out

    assert "Error: LLM unavailable" in output
    renderer.finish.assert_called_once()
    renderer.render_metrics.assert_called_once()


def test_renderer_displays_metrics(capsys) -> None:
    renderer = CLIRenderer()

    metrics = AgentMetrics(
        llm_calls=[
            LLMCallMetrics(
                latency_seconds=0.5,
                input_tokens=150,
                output_tokens=50,
                cached_tokens=25,
            )
        ],
        tool_calls=2,
        tool_errors=1,
    )

    renderer.render_metrics(metrics, context_window=1000)

    output = capsys.readouterr().out

    assert "LLM calls: 1" in output
    assert "Tool calls: 2" in output
    assert "Tool errors: 1" in output
    assert "Input tokens total: 150" in output
    assert "Output tokens total: 50" in output
    assert "Cached tokens total: 25" in output
    assert "Context utilization: 15.0%" in output
