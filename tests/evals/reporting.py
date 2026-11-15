import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from mini_agent.telemetry.telemetry import AgentMetrics

RESULTS_DIR = Path("test-results")


def get_git_sha() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
        timeout=5,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def save_report(
    test_id: str,
    model: str,
    passed: bool,
    metrics: AgentMetrics,
) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    now = datetime.now(UTC)
    timestamp = now.strftime("%Y%m%dT%H%M%S%fZ")

    report = {
        "test_id": test_id,
        "timestamp": now.isoformat(),
        "git_sha": get_git_sha(),
        "model": model,
        "passed": passed,
        "duration_seconds": round(metrics.elapsed_seconds, 3),
        "llm_calls": len(metrics.llm_calls),
        "tool_calls": metrics.tool_calls,
        "tool_errors": metrics.tool_errors,
        "iterations": metrics.iterations,
        "input_tokens": metrics.input_tokens_total,
        "output_tokens": metrics.output_tokens_total,
        "cached_tokens": sum(call.cached_tokens for call in metrics.llm_calls),
    }

    output = RESULTS_DIR / f"{test_id}-{timestamp}.json"

    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return output
