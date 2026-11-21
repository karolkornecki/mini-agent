# Test guide

This supplements the [root guide](../AGENTS.md). Prefer deterministic offline
coverage for implementation changes; no new tests are needed for documentation.

## Commands and categories

Run from the repository root:

```sh
uv run --locked pytest
uv run --locked pytest tests/unit/ -v
uv run --locked pytest tests/integration/ -v
uv run --locked pytest tests/unit/test_read_tools.py -q
```

`pyproject.toml` sets `testpaths = ["tests"]` and
`addopts = "-ra -m 'not live'"`. Integration tests use `@pytest.mark.integration`;
live tests use `@pytest.mark.live` (and may also be integration tests). Selecting
the integration directory keeps the run offline; a marker alone does not imply
that a test is offline.

Live evaluations are opt-in real API calls and can cost money. Only when the task
calls for them, use the README's command:

```sh
RUN_LIVE_TESTS=1 uv run --locked pytest tests/evals/ -m live -v -s
```

Both opt-ins are needed. Live tests load dotenv and require `OPENROUTER_API_KEY`.
`tests/evals/reporting.py` writes metrics JSON to `test-results/`; generated
reports are ignored by git. Do not add real API calls to the default test suite.

## Reusable fixtures and fakes

- `project` in `conftest.py` replaces `mini_agent.tools.paths.ROOT` with `tmp_path`.
  Request it for tests touching file tools; changing the process directory alone
  is insufficient because `ROOT` is captured at import time.
- `sample_project` builds on `project` with `src/main.py`, `src/utils.py`, and
  `README.md`. Keep file writes in temporary workspaces.
- `tests/integration/fakes.py` provides `FakeLLM`, `tool_call_response()`,
  `final_response()`, and `fake_metrics()`. The fake implements `complete_stream`,
  records requests, and consumes scripted responses. Include a final response
  without tool calls to terminate a successful run.
- Patch dependencies where they are used: `agent_trace` on
  `mini_agent.agent.agent`, and `tool_trace` or `get_permission` on
  `mini_agent.agent.tool_execution`. Examples live in
  `integration/test_agent_permissions.py` and `unit/test_tool_execution.py`.
  For SDK stream tests, patch
  `mini_agent.llm.client.llm_trace`. Keep Langfuse and OpenRouter out of offline tests.
- `request_approval()` and `run_cli()` bind `input` as a default argument.
  Pass `ask=` or `read_input=` explicitly in their unit tests. To control the
  agent's approval path, pass `Agent(..., approve=callback)`. Direct
  `execute_tool()` tests supply `approve=` too. Patching `builtins.input` alone
  does not replace the already-bound default.
- Mock `subprocess.run` for command tests and Langfuse's `get_client` for
  observability tests. Use `capsys` for renderer output and collected events
  when testing event emission.
- Agent fakes should return real `LLMCallMetrics`, including
  `usage_available=False` for omitted usage and `finish_reason="length"` for
  incomplete output. `complete_stream` fakes accept keyword-only
  `max_output_tokens`; `FakeLLM` records it with each request. Every completed request emits
  `ContextUsage` after any text deltas and before tool events. Capacity is supplied
  with `Agent(..., context_settings=ContextSettings(...))`; small test windows
  need an explicit smaller output reserve. Renderers read settings from metrics.
  Capacity alerts use separate `ContextWarning` events after `ContextUsage`.

## Where to extend coverage

| Change | Existing tests to inspect |
| --- | --- |
| File access, search, limits, traversal or symlinks | `unit/test_read_tools.py`, `integration/test_workspace.py` |
| File creation, exact replacement, unified diffs | `unit/test_write_tools.py`, `unit/test_apply_patch.py` |
| Tool schemas or registrations | `unit/test_registry.py` |
| Command allowlist, execution, timeouts | `unit/test_run_command.py` |
| Permission policy and user approval | `unit/test_permissions.py`, `unit/test_approval.py`, `integration/test_agent_permissions.py` |
| Tool lifecycle events, results, error accounting, timeout recovery | `unit/test_tool_execution.py`, `integration/test_agent_permissions.py` |
| Conversation history or reset | `unit/test_conversation.py` |
| Streaming text, tool-call fragments, usage | `unit/test_llm_stream.py` |
| Context measurements, unknown usage, task resets and failures | `unit/test_telemetry.py`, `integration/test_context_monitoring.py` |
| Capacity thresholds, warning repetition, output limits | `unit/test_context.py`, `unit/test_llm_stream.py`, `integration/test_context_monitoring.py` |
| Startup context capacity and invalid configuration | `unit/test_main.py` |
| CLI input and event rendering | `unit/test_cli.py`, `unit/test_cli_renderer.py` |
| Metrics or Langfuse behavior | `unit/test_telemetry.py`, `unit/test_observability.py` |

Assert observable results: file contents, absence of writes after rejection,
tool messages and call IDs, emitted events, metrics, and rendered output. For
behavior changes, cover relevant success and failure paths without duplicating
the implementation in the assertions.
