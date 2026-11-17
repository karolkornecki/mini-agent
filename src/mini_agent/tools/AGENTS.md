# Tool implementation guide

This supplements the [root guide](../../../AGENTS.md) for this directory.

## Adding or changing a tool

1. Keep implementation in a small module with a plain function. Use
   `paths.resolve_path()` before file access and UTF-8 for text reads/writes.
2. Register a `Tool` in `registry.py`. `base.py` owns the frozen definition and
   its OpenAI function schema; `params()` sets `additionalProperties: False`.
   Keep property names, required fields, defaults, and types aligned with the
   callable. Schemas describe the model interface; they are not a runtime validator.
3. Add an explicit policy in `../permissions/permissions.py`. A registry entry
   alone does not grant permission: unknown policy entries default to `DENY`.
   Preserve `ASK` for file mutations and commands unless the task changes policy.
4. Update `tests/unit/test_registry.py`'s expected tool names for registry changes,
   permission tests for policy changes, and focused behavior tests. Use fake-LLM
   integration coverage when the agent's execution/approval flow changes.

## Path and output contracts

- Import the `paths` module and access `paths.ROOT` dynamically; tests monkeypatch
  it to isolate workspaces. Do not cache another copy of `ROOT` in tool modules.
- `resolve_path()` resolves symlinks and rejects targets outside `ROOT`. It blocks
  `.git`, `.venv`, `.ssh`, `.aws`, `__pycache__`, `node_modules`, and any path
  component equal to `.env` or starting with `.env.`. The latter also blocks
  `.env.example` through the runtime tools.
- Reuse `MAX_FILE_SIZE` (100,000 bytes) and `MAX_RESULTS` (100). Content limits
  measure UTF-8 bytes. Listing/search tools skip inaccessible entries; direct
  access tools raise errors. Keep output paths relative to the workspace.
- `write_file()` creates new files and parent directories but rejects overwrites.
  `edit_file()` requires nonempty `old_text` occurring exactly once.
  `apply_patch()` accepts a unified diff for one existing file, validates context,
  and rejects creation, deletion, and renaming. Preserve the original file when
  validation fails; check the final size before writing.
- Keep return values JSON-serializable. `agent/tool_execution.py` serializes
  successful results with `json.dumps(..., ensure_ascii=False)` and handles selected exception types
  as tool-error messages. If introducing a new error type, inspect
  [tool execution](../agent/tool_execution.py) to determine how it is handled.

## Commands

`run_command()` accepts an identifier from `ALLOWED_COMMANDS`, never arbitrary
shell input. Current identifiers are `git_status`, `git_diff`, `pytest`, and `ruff`.
The registry derives the command schema enum from that mapping.

Preserve argument-list execution without a shell, `cwd=paths.ROOT`, the 30-second
timeout, captured stdout/stderr, and 10,000-character truncation of each stream.
A nonzero exit code is returned in the result text. The raw tool propagates
subprocess exceptions; `execute_tool()` turns `TimeoutExpired` into a tool-error
result so the model can continue. Command tests mock subprocess execution rather
than running commands inside the test process.

See [the test guide](../../../tests/AGENTS.md) for fixtures and targeted checks.
