# Working on mini-agent

## Project intent

Read [README.md](README.md) before changing the repository. This is a small,
deliberately non-production coding agent built to learn how agents work. Prefer
small, readable changes that make the execution flow easy to follow. Avoid
frameworks, speculative abstractions, and production infrastructure unless the
task calls for them. Commit history documents the learning progression.

These files guide coding agents developing this repository. The mini-agent
application does not currently load `AGENTS.md` into its own conversation.

## Setup and commands

Run commands from the repository root. Use Python 3.12+ and `uv`; `.python-version`
and CI currently select 3.12. Preserve that compatibility even though Ruff's
`target-version` is currently `py313`.

```sh
uv sync --locked --dev
uv run --locked pytest
uv run --locked ruff check .
```

- Offline unit and integration tests need no API keys or `.env` setup.
- For Python edits, check formatting with
  `uv run --locked ruff format --check <changed-python-paths>`. Keep unrelated
  formatting changes out of the task.
- `uv run --locked mini-agent` starts the interactive application and requires
  an OpenRouter API key. It is not an offline smoke test. See the README for setup.
- Live evaluations require both `RUN_LIVE_TESTS=1` and `-m live`; run them only
  when the task explicitly calls for real model requests, which can incur costs.
- CI runs `uv sync --locked --dev` and `uv run --locked pytest` on pull requests.
  The optional `.githooks/pre-commit` hook runs locked Ruff linting. Enable it
  locally with `git config core.hooksPath .githooks` if desired.

## Code map

| Location | Responsibility |
| --- | --- |
| `src/mini_agent/main.py` | Load dotenv, wire the agent and renderer, start the CLI, print metrics and flush traces on exit. |
| `src/mini_agent/cli/cli.py` | Interactive input loop, `/clear`, `/exit`, and task error display. |
| `src/mini_agent/cli/renderer.py` | Render streamed text and tool events in the terminal. |
| `src/mini_agent/agent/agent.py` | System prompt, conversation state, model loop, task metrics, and approval callback wiring. |
| `src/mini_agent/agent/tool_execution.py` | Per-call arguments, registry lookup, permission-checked invocation, tracing, events, tool metrics, and result messages. |
| `src/mini_agent/agent/events.py` | Frozen `TextDelta`, `ToolStarted`, and `ToolFinished` event dataclasses. |
| `src/mini_agent/llm/client.py` | OpenAI SDK client pointed at OpenRouter; reconstruct streamed text/tool calls and collect usage. |
| `src/mini_agent/tools/` | Plain tool functions, shared path checks, schemas, registry, command allowlist. Read [its guide](src/mini_agent/tools/AGENTS.md) when changing tools. |
| `src/mini_agent/permissions/` | `ALLOW`/`ASK`/`DENY` policy and interactive approval. |
| `src/mini_agent/telemetry/` | Per-run metrics and optional Langfuse observation contexts. |
| `tests/` | Unit, fake-LLM integration, and opt-in live tests. Read [its guide](tests/AGENTS.md) when changing tests. |
| `pyproject.toml`, `uv.lock` | Dependencies, packaging, entry point, Ruff and pytest settings. |

The project/console command is `mini-agent`; Python imports use `mini_agent`.

## Behavior to preserve

- `LLMClient.complete_stream(messages, tools, on_text)` returns an OpenAI
  `ChatCompletionMessage` and `LLMCallMetrics`. It forwards text fragments through
  `on_text` and assembles tool-call fragments by index, including arguments and IDs.
  Usage may arrive in a final chunk without choices, or be absent entirely.
- `Agent.run()` resets metrics for each task but retains conversation history
  across successful turns. `_run()` works on a copy and commits it to
  `self.messages` only upon a response without tool calls. `/clear` keeps only the
  system message. The loop currently permits 150 model iterations per task.
- `execute_tool()` returns a matching tool-result message with its `tool_call_id`
  for success, rejection, and handled errors. It owns the started/finished events
  and tool counters; `Agent` appends its returned message. Tool outputs are
  JSON-encoded on success. A rejected approval is `denied` without incrementing
  `tool_errors`; policy denial and handled tool exceptions are `error` and do
  increment it. Handled exceptions are `ValueError`, `TypeError`, `OSError`
  (including `PermissionError`), `KeyError`, and `TimeoutExpired`. Other exceptions
  and event-callback failures propagate.
- `execute_with_permission()` in `agent/tool_execution.py` takes the tool's
  callable and checks policy and approval before tracing and invoking it.
  Policy denial raises `PermissionError`; user rejection raises `ApprovalRejected`,
  which `execute_tool()` translates to the `denied` outcome.
- Keep terminal rendering in `cli/renderer.py` through the event callback.
  Approval prompting lives separately in `permissions/approval.py`; pass
  `approve=` to `Agent` to replace that prompt with a callback.
- File tools use `tools.paths.ROOT`, captured from the working directory at import
  time. Changing directory later does not update it. Preserve shared path checks,
  blocked paths, size limits, and the explicit command allowlist.
- Read tools are allowed; file mutations and commands ask for approval; tools
  without an explicit permission default to denied. This describes mini-agent's
  runtime policy, not an extra approval requirement for developing this repo.
- `MODEL_CONTEXT_WINDOW` affects reported utilization only. Metrics describe the
  latest task, not the whole CLI session. Tracing is optional; preserve tested
  no-op fallback behavior without swallowing exceptions from agent execution.

## Change workflow

Inspect the relevant implementation and tests first. Follow the existing plain
functions, dataclasses, type hints, and module boundaries. Ruff uses a 100-character
line length and rules `E`, `F`, `I`, `B`, and `UP`.

For behavior changes, add or adjust focused tests, then run the offline suite and
lint. Documentation-only changes need link/content checks rather than new tests.
Update the README and these guides when commands, architecture, or contracts
change. Keep `pyproject.toml` and `uv.lock` consistent when changing dependencies;
do not hand-edit the lockfile.

Use `.env.example` to understand configuration. Do not inspect or expose actual
`.env` files or credentials, and do not overwrite a developer's environment file.
Do not commit generated reports, caches, or virtual environments. In the handoff,
state what changed, what was checked, and any failures or unverified behavior.
