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

The task is currently defined directly in `src/mini_agent/main.py`, in the `agent.run(...)` call. Change that text to try a different task. File tools operate relative to the directory where you start the program. The agent asks for approval before creating or editing files or running commands.

## Configuration

The application loads settings from `.env`. The example file contains these settings:

| Variable | Purpose | Value in `.env.example` |
| --- | --- | --- |
| `OPENROUTER_API_KEY` | Required API key for model requests through OpenRouter. | Placeholder; replace with your key. |
| `MODEL` | OpenRouter model identifier. | `qwen/qwen3-coder` |
| `MODEL_CONTEXT_WINDOW` | Context window size in tokens, used to calculate utilization in the metrics report. It does not change the model's actual limit. | `128000` |
| `LANGFUSE_TRACING_ENABLED` | Enable or disable Langfuse tracing. | `false` |
| `LANGFUSE_PUBLIC_KEY` | Public key for your Langfuse project. | Commented out. |
| `LANGFUSE_SECRET_KEY` | Secret key for your Langfuse project. | Commented out. |
| `LANGFUSE_BASE_URL` | URL of your Langfuse instance. | Commented out; example uses `https://cloud.langfuse.com`. |

To enable tracing, set `LANGFUSE_TRACING_ENABLED=true`, uncomment the Langfuse settings, and fill in your project keys and instance URL.

## How it works

The agent sends a task, conversation history, and available tool definitions to a model through OpenRouter. When the model requests tools, the agent executes them and adds their results to the conversation. This loop continues until the model returns a response without tool calls, or the agent reaches its iteration limit.

The tools support listing, reading, and searching files; creating and editing files; applying patches; and running a small set of predefined commands. Metrics track model latency, token usage, iterations, tool calls, and errors. Optional Langfuse traces record agent runs, model calls, and tool execution.

## Directory structure

The main files and directories are:

```text
mini-agent/
├── src/
│   └── mini_agent/
│       ├── main.py               # Entry point, example task, and final metrics output
│       ├── agent/
│       │   └── agent.py          # Conversation loop, tool execution, and approval prompts
│       ├── llm/
│       │   └── client.py         # OpenRouter requests and model usage metrics
│       ├── tools/
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
├── tests/                       # Tests for tools, registration, and metrics
├── .env.example                 # Environment configuration template
├── pyproject.toml               # Package metadata, dependencies, and tool settings
├── uv.lock                      # Locked dependency versions
└── README.md                    # Project purpose and local setup
```

## Development

Run the tests and lint checks with:

```sh
uv run pytest
uv run ruff check .
```
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
