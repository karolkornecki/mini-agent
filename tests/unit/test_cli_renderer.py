from mini_agent.agent.events import (
    TextDelta,
    ToolFinished,
    ToolStarted,
)
from mini_agent.cli.renderer import CLIRenderer


def test_text_streaming(capsys):
    renderer = CLIRenderer()

    renderer.render(TextDelta("Hello "))
    renderer.render(TextDelta("world"))
    renderer.finish()

    assert capsys.readouterr().out == "Hello world\n"


def test_tool_execution(capsys):
    renderer = CLIRenderer()

    renderer.render(
        ToolStarted(
            call_id="call_1",
            name="read_file",
            arguments={"path": "README.md"},
        )
    )

    renderer.render(
        ToolFinished(
            call_id="call_1",
            name="read_file",
            status="success",
            output="File contents",
        )
    )

    output = capsys.readouterr().out

    assert "→ read_file" in output
    assert "README.md" in output
    assert "✓ read_file [success]" in output
    assert "File contents" not in output


def test_tool_error(capsys):
    renderer = CLIRenderer()

    renderer.render(
        ToolFinished(
            call_id="call_1",
            name="read_file",
            status="error",
            output="File not found",
        )
    )

    output = capsys.readouterr().out

    assert "✗ read_file [error]" in output
    assert "File not found" in output
