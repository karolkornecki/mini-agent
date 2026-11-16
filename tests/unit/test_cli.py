from unittest.mock import Mock

from mini_agent.cli.cli import run_cli


def test_cli_runs_multiple_tasks() -> None:
    agent = Mock()
    renderer = Mock()
    agent.run.return_value = "Done"

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


def test_cli_ignores_empty_input() -> None:
    agent = Mock()
    renderer = Mock()

    commands = iter(["", "   ", "/exit"])

    run_cli(
        agent,
        renderer,
        read_input=lambda _: next(commands),
    )

    agent.run.assert_not_called()


def test_cli_clears_conversation() -> None:
    agent = Mock()
    renderer = Mock()

    agent.run.return_value = "Done"

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
    agent.run.assert_any_call("read README.md")
    agent.run.assert_any_call("list files")
