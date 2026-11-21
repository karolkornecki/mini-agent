# mini-agent

A small coding agent built from scratch for learning purposes.

I’m building this project to understand how coding agents like OpenCode work internally. For me, the best way to understand them is to implement a simple version myself, experiment with it, and learn along the way.

**This project is not production ready and never will be.** It is a personal learning exercise, with no plans to turn it into a production tool. You can explore the experiments and understand the purpose behind them.

To follow the learning process, explore the repository's commit history from the earliest commits onward. Reading the changes commit by commit can help you understand how the agent grows from its simplest form, with each addition building on what came before.

## Running locally

You need Python 3.12 or newer and [uv](https://docs.astral.sh/uv/).

Install dependencies and create your local environment file:

```sh
uv sync
cp .env.example .env
```

Edit `.env` before running:

- Set `OPENROUTER_API_KEY` to your OpenRouter API key.
- Set `MODEL` to the OpenRouter model you want to use, or keep `qwen/qwen3-coder`.
- Set `MODEL_CONTEXT_WINDOW` to your selected model's context window size in tokens so the metrics report can calculate utilization correctly.
- Set `LANGFUSE_TRACING_ENABLED` to `false` to run without tracing, or `true` to enable it. When enabled, uncomment and set `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and `LANGFUSE_BASE_URL` for your Langfuse project.

See [Configuration](#configuration) below for details. Then run from the repository root:

```sh
uv run mini-agent
```

You can also run the Python module with `uv run python -m mini_agent.main`.
The project and command are named `mini-agent`; the Python package is `mini_agent`.

Enter a task at the interactive `>` prompt. The agent streams its response and
shows tool activity and context usage in the terminal. Conversation history is
retained between tasks; use `/clear` to reset it and `/exit` to quit. Metrics for
the latest task are printed after each task, including failed tasks.

File tools operate relative to the directory where you start the program. The
agent asks for approval before creating or editing files or running commands.

## Configuration

The application loads settings from `.env`. The example file contains these settings:

| Variable | Purpose | Value in `.env.example` |
| --- | --- | --- |
| `OPENROUTER_API_KEY` | Required API key for model requests through OpenRouter. | Placeholder; replace with your key. |
| `MODEL` | OpenRouter model identifier. | `qwen/qwen3-coder` |
| `MODEL_CONTEXT_WINDOW` | Positive integer context capacity in tokens, used for per-request utilization and the task report. Defaults to `128000` when absent; invalid values stop startup. It does not change the model's actual limit. | `128000` |
| `LANGFUSE_TRACING_ENABLED` | Enable or disable Langfuse tracing. | `false` |
| `LANGFUSE_PUBLIC_KEY` | Public key for your Langfuse project. | Commented out. |
| `LANGFUSE_SECRET_KEY` | Secret key for your Langfuse project. | Commented out. |
| `LANGFUSE_BASE_URL` | URL of your Langfuse instance. | Commented out; example uses `https://cloud.langfuse.com`. |

To enable tracing, set `LANGFUSE_TRACING_ENABLED=true`, uncomment the Langfuse settings, and fill in your project keys and instance URL.

## How it works

The agent sends a task, conversation history, and available tool definitions to a model through OpenRouter. When the model requests tools, the agent executes them and adds their results to the conversation. This loop continues until the model returns a response without tool calls, or the agent reaches its iteration limit.

The tools support listing, reading, and searching files; creating and editing files; applying patches; and running a small set of predefined commands. Metrics track model latency, token usage, iterations, tool calls, and errors. Optional Langfuse traces record agent runs, model calls, and tool execution.

After every completed model request, the CLI reports the provider's input-token
count divided by the configured context capacity. For example:

```text
Context (last request, iteration 1): 12,800 / 128,000 tokens (10.0%)
```

This estimates the context sent in that request, including conversation history
and tool definitions. It excludes the generated response and any later tool
results. Cumulative input-token totals count usage across requests and are
reported separately. Percentages above 100% are displayed as calculated.

When the provider omits usage, that request's context is `unknown`; a reported
zero is `0.0%`. Token totals sum the usage received, with missing usage contributing
zero. A failed request produces no context measurement, so the task report uses
the latest completed request, or `unknown` if none completed. Each task starts
with fresh metrics; `/clear` resets conversation history only.

`AgentMetrics(context_window=...)` exposes `context_input_tokens` and
`context_utilization_percent` for the latest request, returning `None` when
unknown. The existing `llm_calls` list retains per-request token counts and
`usage_available`. `Agent` emits a `ContextUsage` event after recording each
request, before executing its tool calls; the CLI renderer handles its display.

`Agent` owns the conversation and model loop. For each tool call, it delegates to
`execute_tool()`, which parses arguments, delegates permission checks and invocation
to `execute_with_permission()`, emits lifecycle events, updates tool metrics, and
returns the complete tool-result message. `execute_with_permission()` takes the
tool's callable, checks policy, requests approval when needed, and traces the
permitted execution. Approval is a callback supplied to `Agent`; it defaults to
the terminal prompt. Command timeouts become tool-error results
so the model can continue. Other unhandled exceptions and event-callback failures
propagate. [GLOSSARY.md](GLOSSARY.md) defines the tool execution and context terms.

## Directory structure

The main files and directories are:

```text
mini-agent/
├── src/
│   └── mini_agent/
│       ├── main.py               # Entry point, CLI wiring, and final metrics output
│       ├── agent/
│       │   ├── agent.py          # Conversation history, model loop, and task metrics
│       │   ├── tool_execution.py # Tool lifecycle, permission checks, events, and results
│       │   └── events.py         # Streamed text, context usage, and tool lifecycle events
│       ├── cli/
│       │   ├── cli.py            # Interactive tasks, /clear, and /exit
│       │   └── renderer.py       # Terminal rendering for agent events
│       ├── llm/
│       │   └── client.py         # OpenRouter streaming and model usage metrics
│       ├── permissions/
│       │   ├── permissions.py    # Allow, ask, or deny policy for each tool
│       │   └── approval.py       # Interactive approval prompts
│       ├── tools/
│       │   ├── AGENTS.md         # Tool development guidance
│       │   ├── base.py           # Tool definition and schema generation
│       │   ├── registry.py       # Available tools and their argument schemas
│       │   ├── paths.py          # Project path checks and file access limits
│       │   ├── list_files.py     # Directory listing
│       │   ├── read_file.py      # Text file reading
│       │   ├── glob.py           # File discovery by pattern
│       │   ├── grep.py           # File content search
│       │   ├── write_file.py     # New file creation
│       │   ├── edit_file.py      # Exact text replacement
│       │   ├── apply_patch.py    # Unified diff application
│       │   └── run_command.py    # Predefined project commands
│       └── telemetry/
│           ├── telemetry.py     # Metrics collection and reporting
│           └── observability.py # Langfuse tracing and flushing
├── tests/                       # Unit, fake-LLM integration, and opt-in live tests
│   └── AGENTS.md                 # Test fixtures, fakes, and coverage map
├── AGENTS.md                     # Repository guidance for coding agents
├── GLOSSARY.md                   # Tool execution and context terminology
├── .env.example                 # Environment configuration template
├── pyproject.toml               # Package metadata, dependencies, and tool settings
├── uv.lock                      # Locked dependency versions
└── README.md                    # Project purpose and local setup
```

## Development

Coding agents should start with [AGENTS.md](AGENTS.md) for the code map, behavior
contracts, and development workflow. Additional guides cover
[tool development](src/mini_agent/tools/AGENTS.md) and
[testing](tests/AGENTS.md). These files describe how to work on the repository;
the mini-agent application does not currently load them into its own prompts.

Run the tests and lint checks with:

```sh
uv run pytest
uv run ruff check .
```

Enable the pre-commit hook after cloning:

```sh
git config core.hooksPath .githooks
```

The hook runs `uv run --locked ruff check .` before every commit and blocks the
commit if linting fails. It checks the working tree without changing files.

## Running Tests

The project uses **pytest** for automated testing and **uv** for dependency management.

Tests are divided into three categories:

| Type | Directory | Description |
|---|---|---|
| Unit | `tests/unit/` | Tests individual components in isolation |
| Integration | `tests/integration/` | Tests agent loop, tools, registry, and permissions using a Fake LLM |
| Live / Evals | `tests/evals/` | Tests the agent against a real LLM via OpenRouter |

### Run All Tests (Except Live)

```bash
uv run pytest -v
```

Live tests are excluded by default to prevent accidental API usage and costs.

GitHub Actions runs the unit and integration tests whenever a pull request is opened,
updated, or reopened. The workflow uses the Python version in `.python-version` and
installs dependencies from `uv.lock`. Live tests remain excluded.

### Run Unit Tests

```bash
uv run pytest tests/unit/ -v
```

### Run Integration Tests

```bash
uv run pytest tests/integration/ -v
```

Integration tests use a Fake LLM and isolated temporary workspaces. They do not require an OpenRouter API key.

### Run Live Tests

Live tests execute the agent against a real LLM and require a valid `OPENROUTER_API_KEY` in the `.env` file.

```bash
RUN_LIVE_TESTS=1 uv run pytest tests/evals/ -m live -v -s
```

Run a specific live test:

```bash
RUN_LIVE_TESTS=1 uv run pytest \
  tests/evals/test_live_read.py \
  -m live -v -s
```

The `-m live` option is required because the default pytest configuration excludes tests marked as `live`.

The `RUN_LIVE_TESTS=1` environment variable explicitly enables real API calls.

**Warning:** Live tests consume API tokens, may incur costs, and can produce nondeterministic results.

### Test Reports

Generate a JUnit XML report:

```bash
uv run pytest --junitxml=test-results/results.xml
```

Run tests with detailed output:

```bash
uv run pytest -vv -s
```

### Code Quality

Run Ruff linting:

```bash
uv run ruff check .
```

Check formatting:

```bash
uv run ruff format --check .
```

### PyCharm Configuration

For live tests, configure the pytest run configuration with:

- **Additional Arguments:** `-m live -v -s`
- **Environment Variables:** `RUN_LIVE_TESTS=1`

The default pytest configuration excludes live tests:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra -m 'not live'"

markers = [
    "integration: integration tests",
    "live: tests requiring real LLM API",
]
```
