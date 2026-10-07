# Buddy-MLIR agent instructions

Buddy-MLIR is an MLIR-based compiler framework from DSLs to domain-specific
architectures. It also builds LLMs (DeepSeek R1, Qwen, Whisper, ...) into
`.rax` files that `buddy-cli` runs on CPUs (x86, RVV), the SpacemiT K3 and
Tenstorrent devices.

These rules hold for every task. Task-specific workflows are skills in
`.agents/skills/` (also visible to Claude Code through `.claude/skills/`).

## Build and test

```bash
# LLVM/MLIR (submodule llvm/): see README.md, "Build and Test LLVM/MLIR/CLANG"
cmake -G Ninja -S . -B build \
    -DMLIR_DIR=$PWD/llvm/build/lib/cmake/mlir \
    -DLLVM_DIR=$PWD/llvm/build/lib/cmake/llvm \
    -DLLVM_ENABLE_ASSERTIONS=ON -DCMAKE_BUILD_TYPE=RELEASE \
    -DBUDDY_MLIR_ENABLE_PYTHON_PACKAGES=ON \
    -DPython3_EXECUTABLE="$(which python)"
ninja -C build
ninja -C build check-buddy        # lit tests: run before every PR
```

Models: `python3 tools/buddy-codegen/build_model.py --spec <spec.json>
--build-dir build ...` (README.md, docs/CrossCompilingRaxFile.md,
docs/K3DeepSeekR1.md).

- `tools/buddy-codegen/import_model.py` imports `buddy.compiler` from
  `<repo>/build/python_packages`, whatever `--build-dir` is. After changing
  Python under `frontend/`, rebuild `build` (`ninja -C build`) and delete
  `<build-dir>/models/<model>/.buddy_import_done` before rebuilding a model.

## Rules

- Correctness first. Say whether a change keeps results bit for bit or
  changes the numerics; a change of numerics needs a measured accuracy check
  (see the `performance-optimization` skill).
- Small, independently reviewable patches; no unrelated refactoring in an
  optimization PR. Update the docs and tests the change touches.
- Do not modify the `llvm` submodule unless the task requires it.
- Style: clang-format (C/C++) and ruff (Python) through pre-commit. Run it on
  the changed files only (`pre-commit run --files ...`); `--all-files`
  rewrites unrelated files.
- Never commit secrets, host names, accounts or local paths of a developer
  machine or board.

## Skills

| Task | Skill |
| --- | --- |
| Making anything faster (kernels, runtime, generated code), comparing against another framework | `performance-optimization` |
| LLM prefill / decode / KV cache / sampling performance, perplexity | `llm-inference-optimization` |
| RVV code generation: LMUL, register pressure, instruction choice | `riscv-rvv-optimization` |
| Facts about a specific chip (SpacemiT K3, Tenstorrent, ...) | `hardware-targets` |
| Preparing a commit or pull request | `upstream-pr` |
