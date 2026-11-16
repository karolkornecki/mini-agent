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


def test_empty_metrics() -> None:
    metrics = AgentMetrics()

    report = metrics.report()

    assert "LLM calls: 0" in report
    assert "Input tokens total: 0" in report
    assert "Cached tokens total: 0" in report
    assert "Context utilization: 0.0%" in report

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

    report = metrics.report(context_window=1000)

    assert "LLM calls: 1" in report
    assert "Tool calls: 2" in report
    assert "Tool errors: 0" in report
    assert "Input tokens total: 100" in report
    assert "Output tokens total: 20" in report
    assert "Cached tokens total: 40" in report
    assert "Context utilization: 10.0%" in report
