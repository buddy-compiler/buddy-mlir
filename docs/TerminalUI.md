# Terminal UI Helpers

`tools/buddy/utils/terminal.py` provides shared terminal UI helpers for Python
tools.  Use it when a tool needs consistent section headers, command progress,
spinner animation, elapsed time, and color handling.

The default visual style uses an orange accent for active progress and gray for
status metadata.  It automatically falls back to plain text when stdout is not a
TTY.

## Import

Build `buddy-mlir` with Python packages enabled and add the generated package
directory to `PYTHONPATH`:

```bash
export BUDDY_MLIR_BUILD_DIR=/path/to/buddy-mlir/build
export PYTHONPATH=${BUDDY_MLIR_BUILD_DIR}/python_packages:${PYTHONPATH}
```

Then import directly:

```python
from buddy.utils.terminal import TerminalUI, run_with_status
```

## Sections

Use `TerminalUI.section()` for major phases:

```python
ui = TerminalUI()

ui.section("ModelBench Build: deepseek-r1")
ui.section("ModelBench Run: deepseek-r1")
```

Interactive terminals render the section title with the shared orange accent.

## Running Commands

Use `run_with_status()` to run a subprocess with spinner animation and a final
status line:

```python
result = run_with_status(
    ["python3", "script.py"],
    cwd=ROOT,
    ui=ui,
    label="import model",
    phase="build",
    capture_stdout=True,
)

output = result.stdout.decode(errors="replace")
```

For commands that write generated files, pass `stdout=Path(...)`:

```python
run_with_status(
    ["buddy-opt", "input.mlir", "-canonicalize"],
    cwd=ROOT,
    ui=ui,
    stdout=out_dir / "canonical.mlir",
    label="canonicalize MLIR",
    phase="build",
)
```

Set `verbose=True` to print the full command before running it:

```python
run_with_status(cmd, cwd=ROOT, ui=ui, label="link runner", verbose=True)
```

On failure, `run_with_status()` raises `RuntimeError` with the exit code, full
command, and captured stdout/stderr.

## Tables And Timings

Use `TerminalUI.format_seconds()` for consistent elapsed-time formatting:

```python
print(ui.format_seconds(0.0311008))  # 31.101 ms
```

Use `TerminalUI.rule(width)` for dim table separators:

```python
print(f"{'name':<16} {'time':>12}")
print(ui.rule(30))
```

## Environment Variables

`NO_COLOR=1` disables ANSI colors.

`NO_ANIMATION=1` disables the spinner and prints plain start/done lines.

When stdout is redirected or not a TTY, animation and color are disabled
automatically.

## Recommended Pattern

Keep tool-specific logic in the tool script and use `TerminalUI` only for
presentation:

```python
ui = TerminalUI()

ui.section("Build")
run_with_status(build_cmd, cwd=ROOT, ui=ui, label="compile")

ui.section("Run")
result = run_with_status(
    run_cmd,
    cwd=ROOT,
    ui=ui,
    label="execute",
    phase="run",
    capture_stdout=True,
)
```

This keeps all Python tools visually consistent while preserving plain,
script-friendly output in non-interactive environments.
