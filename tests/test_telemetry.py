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

    report = metrics.report(context_window=128000)

    assert "Context utilization: 50.0%" in report
    assert "Input tokens total: 65000" in report


def test_empty_metrics():
    metrics = AgentMetrics()

    assert metrics.input_tokens_total == 0
    assert metrics.output_tokens_total == 0
    assert "Context utilization: 0.0%" in (metrics.report(128000))


def test_elapsed_time_non_negative():
    metrics = AgentMetrics()
    assert metrics.elapsed_seconds >= 0
