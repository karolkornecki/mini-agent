import pytest

from mini_agent.context import ContextLevel, ContextSettings
from mini_agent.telemetry.telemetry import (
    AgentMetrics,
    LLMCallMetrics,
)


def test_llm_token_totals():
    metrics = AgentMetrics(
        llm_calls=[
            LLMCallMetrics(1.2, 1000, 200, 100),
            LLMCallMetrics(1.5, 2000, 300, 500),
        ]
    )

    assert metrics.input_tokens_total == 3000
    assert metrics.output_tokens_total == 500


def test_context_utilization():
    metrics = AgentMetrics(
        llm_calls=[
            LLMCallMetrics(1.0, 1000, 100),
            LLMCallMetrics(1.0, 64000, 100),
        ]
    )

    report = metrics.report()

    assert metrics.context_input_tokens == 64000
    assert metrics.context_utilization_percent == 50.0
    assert "Context utilization (last request): 50.0%" in report
    assert "Input tokens total: 65000" in report


def test_empty_metrics() -> None:
    metrics = AgentMetrics()

    report = metrics.report()

    assert "LLM calls: 0" in report
    assert "Input tokens total: 0" in report
    assert "Cached tokens total: 0" in report
    assert metrics.context_input_tokens is None
    assert metrics.context_utilization_percent is None
    assert "Context utilization (last request): unknown" in report


@pytest.mark.parametrize(
    ("input_tokens", "expected_percent"),
    [(0, 0.0), (125, 12.5), (1500, 150.0)],
)
def test_context_utilization_uses_configured_capacity(input_tokens, expected_percent) -> None:
    metrics = AgentMetrics(
        context_settings=ContextSettings(context_window=1000, output_reserve=100),
        llm_calls=[LLMCallMetrics(1.0, input_tokens, 500, cached_tokens=input_tokens)],
    )

    assert metrics.context_input_tokens == input_tokens
    assert metrics.context_utilization_percent == expected_percent
    assert f"Context utilization (last request): {expected_percent:.1f}%" in metrics.report()


def test_missing_latest_usage_does_not_reuse_previous_measurement() -> None:
    metrics = AgentMetrics(
        context_settings=ContextSettings(context_window=1000, output_reserve=100),
        llm_calls=[
            LLMCallMetrics(1.0, 500, 100),
            LLMCallMetrics(1.0, 0, 0, usage_available=False),
        ],
    )

    assert metrics.context_input_tokens is None
    assert metrics.context_utilization_percent is None
    assert metrics.input_tokens_total == 500
    assert metrics.output_tokens_total == 100
    assert "Context utilization (last request): unknown" in metrics.report()


def test_context_state_uses_latest_request_and_retains_previous_snapshot() -> None:
    settings = ContextSettings(context_window=1200, output_reserve=200)
    metrics = AgentMetrics(context_settings=settings, llm_calls=[LLMCallMetrics(0.1, 800, 20)])
    previous = metrics.context_state

    metrics.llm_calls.append(LLMCallMetrics(0.1, 500, 20))

    assert previous.settings is settings
    assert previous.level == ContextLevel.WARNING
    assert previous.remaining_input_tokens == 200
    assert metrics.context_state.settings is settings
    assert metrics.context_state.level == ContextLevel.NORMAL
    assert metrics.context_state.remaining_input_tokens == 500
    assert metrics.input_tokens_total == 1300


def test_elapsed_time_non_negative():
    metrics = AgentMetrics()
    assert metrics.elapsed_seconds >= 0


def test_metrics_totals() -> None:
    metrics = AgentMetrics(
        llm_calls=[
            LLMCallMetrics(
                latency_seconds=1.0,
                input_tokens=100,
                output_tokens=20,
                cached_tokens=40,
            ),
            LLMCallMetrics(
                latency_seconds=2.0,
                input_tokens=200,
                output_tokens=30,
                cached_tokens=80,
            ),
        ],
        iterations=2,
        tool_calls=3,
        tool_errors=1,
    )

    assert metrics.input_tokens_total == 300
    assert metrics.output_tokens_total == 50
    assert metrics.cached_tokens_total == 120
    assert metrics.tool_calls == 3
    assert metrics.tool_errors == 1


def test_metrics_report() -> None:
    metrics = AgentMetrics(
        context_settings=ContextSettings(context_window=1000, output_reserve=100),
        llm_calls=[
            LLMCallMetrics(
                latency_seconds=1.0,
                input_tokens=100,
                output_tokens=20,
                cached_tokens=40,
            )
        ],
        iterations=1,
        tool_calls=2,
        tool_errors=0,
    )

    report = metrics.report()

    assert "LLM calls: 1" in report
    assert "Tool calls: 2" in report
    assert "Tool errors: 0" in report
    assert "Input tokens total: 100" in report
    assert "Output tokens total: 20" in report
    assert "Cached tokens total: 40" in report
    assert "Context utilization (last request): 10.0%" in report
    assert "Input capacity: 900 tokens (100 reserved for output)" in report
    assert "Input capacity utilization (last request): 11.1%" in report
    assert "Remaining input capacity (last request): 800 tokens" in report
    assert "Context level (last request): normal" in report
