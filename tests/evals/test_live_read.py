import os
from pathlib import Path
from typing import Any

import pytest
from dotenv import load_dotenv

import mini_agent.agent.tool_execution as execution_module
from mini_agent.agent.agent import Agent
from mini_agent.llm.client import LLMClient
from tests.evals.reporting import save_report

load_dotenv()


@pytest.mark.live
@pytest.mark.integration
def test_live_read_file(
    project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if os.getenv("RUN_LIVE_TESTS") != "1":
        pytest.skip("Live tests require RUN_LIVE_TESTS=1")

    if not os.getenv("OPENROUTER_API_KEY"):
        pytest.skip("Missing OPENROUTER_API_KEY")

    (project / "answer.txt").write_text(
        "The secret test number is 73921.",
        encoding="utf-8",
    )

    # Restrict this test to a read-only tool.
    import importlib

    agent_module = importlib.import_module(Agent.__module__)

    schemas = [
        schema
        for schema in agent_module.get_tool_schemas()
        if schema["function"]["name"] == "read_file"
    ]

    monkeypatch.setattr(
        agent_module,
        "get_tool_schemas",
        lambda: schemas,
    )

    # No interactive approvals.
    def fail_approval(name: str, arguments: dict[str, Any]) -> bool:
        raise AssertionError(f"Unexpected approval: {name}")

    # Defense in depth: reject every tool except read_file.
    from mini_agent.permissions.permissions import Permission

    monkeypatch.setattr(
        execution_module,
        "get_permission",
        lambda name: Permission.ALLOW if name == "read_file" else Permission.DENY,
    )

    # Construct LLMClient using existing project configuration.
    llm = LLMClient()
    agent = Agent(llm=llm, approve=fail_approval)

    passed = False

    try:
        result = agent.run(
            "Read answer.txt using the read_file tool. Report the exact test number from the file."
        )

        assert "73921" in result
        assert agent.metrics.tool_calls >= 1
        assert agent.metrics.tool_errors == 0
        assert len(agent.metrics.llm_calls) >= 2

        passed = True

    finally:
        report_path = save_report(
            test_id="live-read",
            model=llm.model,
            passed=passed,
            metrics=agent.metrics,
        )
        print(f"\nReport saved: {report_path}")
