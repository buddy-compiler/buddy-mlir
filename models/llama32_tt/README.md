# Llama 3.2 3B Tenstorrent buddy-cli Path

This target builds self-contained `.rax` packages for the native Tenstorrent
`buddy-cli` runtime.

## Shell Variables

Complete [TenstorrentEnvironment.md](../../docs/TenstorrentEnvironment.md)
first.

```bash
cd /path/to/buddy-mlir

export BUDDY_REPO_ROOT=$(pwd)
export BUDDY_BUILD="$BUDDY_REPO_ROOT/build-tenstorrent"
export TTMLIR_BUILD="$BUDDY_REPO_ROOT/build-ttmlir"
export TTMLIR_TOOLCHAIN_DIR="$BUDDY_REPO_ROOT/build-ttmlir-toolchain"
```

## Python Dependencies

Install the model frontend dependencies.

```bash
cd "$BUDDY_REPO_ROOT/thirdparty/tt-mlir"
source env/activate
cd "$BUDDY_REPO_ROOT"

"$TTMLIR_TOOLCHAIN_DIR/venv/bin/python" -m pip install --index-url https://download.pytorch.org/whl/cpu torch
"$TTMLIR_TOOLCHAIN_DIR/venv/bin/python" -m pip install \
  transformers accelerate safetensors sentencepiece
"$TTMLIR_TOOLCHAIN_DIR/venv/bin/python" -c "import torch, transformers; print('llama frontend deps ok')"
```

## Build Batch 1

Build the canonical package.

```bash
cmake -S "$BUDDY_REPO_ROOT" -B "$BUDDY_BUILD" \
  -DBUDDY_BUILD_LLAMA32_TT_MODEL=ON \
  -DBUDDY_LLAMA32_MODEL_PATH=/path/to/Llama-3.2-3B
cmake --build "$BUDDY_BUILD" --target buddy-cli llama32_tt_rax
```

Output:

```text
$BUDDY_BUILD/models/llama32_tt/llama32_tt.rax
```

## Build Fixed Batch

Build a fixed-batch package.

```bash
cmake -S "$BUDDY_REPO_ROOT" -B "$BUDDY_BUILD" \
  -DBUDDY_BUILD_LLAMA32_TT_MODEL=ON \
  -DBUDDY_LLAMA32_MODEL_PATH=/path/to/Llama-3.2-3B \
  -DBUDDY_LLAMA32_FIXED_BATCH_SIZES="32"
cmake --build "$BUDDY_BUILD" --target llama32_tt_b32_rax
```

Output:

```text
$BUDDY_BUILD/models/llama32_tt_b32/llama32_tt_b32.rax
```

## Run Setup

Set runtime environment variables.

```bash
source "$BUDDY_REPO_ROOT/thirdparty/tt-mlir/env/activate"
cd "$BUDDY_REPO_ROOT"

ulimit -v 95000000
ulimit -m 95000000
export TT_METAL_RUNTIME_ROOT="$BUDDY_REPO_ROOT/thirdparty/tt-mlir/third_party/tt-metal/src/tt-metal"
export TT_METAL_HOME="$TT_METAL_RUNTIME_ROOT"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$TTMLIR_BUILD/python_packages:${PYTHONPATH:-}"
export BUDDY_RAX_PAYLOAD_DIR=/tmp/$USER/buddy_rax_payload
mkdir -p "$BUDDY_RAX_PAYLOAD_DIR"
```

## Run Batch 1

Run the canonical package.

```bash
"$BUDDY_BUILD/bin/buddy-cli" \
  --model "$BUDDY_BUILD/models/llama32_tt/llama32_tt.rax" \
  --prompt "I like taking walks in the" \
  --max-tokens 32 \
  --temperature 0
```

## Run Fixed Batch

Run a fixed-batch package.

```bash
"$BUDDY_BUILD/bin/buddy-cli" \
  --model "$BUDDY_BUILD/models/llama32_tt_b32/llama32_tt_b32.rax" \
  --prompt-file "$BUDDY_REPO_ROOT/models/llama32_tt/llama32_3b_default_prompts.txt" \
  --prompt-length 32 \
  --batch-size 32 \
  --max-tokens 96 \
  --print-all-batch \
  --temperature 0
```

## Basic Settings

- Llama 3.2 uses plain prompt tokenization by default. Pass `--chat-template` to use chat formatting.
- The default cache length is 128 tokens.
- `--prompt-file` accepts one prompt or exactly `--batch-size` prompts.
- Batch output prints user 0 by default. Use `--print-all-batch` for all users.
- EOS/EOT stopping is enabled by default.
- Set `BUDDY_LLAMA31_IGNORE_EOS=1` before packaging only for fixed-length benchmarks.
- Set `TT_LOGGER_LEVEL=FATAL`, `TT_METAL_LOGGER_LEVEL=FATAL`, and `TTMLIR_RUNTIME_LOGGER_LEVEL=FATAL` to suppress runtime logs.

## Runtime Options

Model source:

- `--model <path.rax>` selects the self-contained `.rax` package.

Prompt input:

- `--prompt <text>` passes one prompt on the command line.
- `--prompt-file <path>` loads one prompt per line for fixed-batch runs.
- `--prompt-length <N>` sets the fixed prefill length in tokens. Prompts longer
  than this fail; shorter prompts are padded for fixed-batch runs.
- If neither `--prompt` nor `--prompt-file` is provided, `buddy-cli` reads a
  prompt from stdin.

Batching:

- Batch size is fixed by the `.rax` package shape.
- `--batch-size <N>` may be passed for clarity, but it must match the package.
- `--prompt-file` may contain one prompt, which is broadcast to the full batch,
  or exactly `--batch-size` prompts.
- `--print-all-batch` prints all batch outputs. Without it, batch runs print
  user 0 only.

Generation length and stopping:

- `--max-tokens <N>` caps generated tokens.
- Generation also stops on EOS/EOT unless the package was built with
  `BUDDY_LLAMA31_IGNORE_EOS=1`.
- Generation cannot exceed the package cache length. For the default Llama 3.2
  packages, the cache length is 128 tokens.

Sampling:

- `--temperature 0` uses greedy decoding.
- `--temperature <float>` enables sampling when greater than zero.
- `--top-k <N>` keeps only the top `N` candidate tokens. `0` disables it.
- `--top-p <float>` enables nucleus sampling. `1.0` disables it.
- `--min-p <float>` enables min-p filtering. `0.0` disables it.
- `--repeat-penalty <float>` penalizes repeated tokens. `1.0` disables it.
- `--repeat-last-n <N>` sets the repetition penalty window.
- `--seed <N>` sets the sampling seed. `0` uses a random seed.

Prompt formatting:

- Llama 3.2 uses plain prompt tokenization by default.
- `--chat-template <path>` loads a JSON chat template override.
- `--interactive` starts a REPL-style session. In this mode, `--prompt` is used
  as the system prompt.

Output and profiling:

- `--no-stats` suppresses timing statistics.
- `--stream-jsonl` emits one JSON object per generated token on stdout. Batch
  runs emit events for every user.
- `--defer-decode-token-readback` defers per-step token readback when supported.
  It requires greedy decoding, device token chaining, persistent decode KV
  reuse, and a package built with `BUDDY_LLAMA31_IGNORE_EOS=1`.
  It cannot be combined with `--stream-jsonl`.
- `BUDDY_LLAMA31_BATCH_TRACE_OUT=<path>` writes a JSON trace for batch runs.

Shared runner environment:

- `BUDDY_RAX_PAYLOAD_DIR=<path>` selects the extracted `.rax` payload cache.
- `BUDDY_LLAMA31_PRINT_ALL_BATCH=1` prints all batch outputs.
- `BUDDY_LLAMA31_DISABLE_DEVICE_TOKEN_CHAIN=1` disables device-side token
  chaining for debug comparison.
- `BUDDY_LLAMA31_RETAIN_DECODE_LOGITS=1` keeps decode logits for debug
  inspection.
