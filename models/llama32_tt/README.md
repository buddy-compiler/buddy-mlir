# Llama 3.2 3B Tenstorrent buddy-cli Path

This target builds self-contained `.rax` packages for `meta-llama/Llama-3.2-3B`
using the native Tenstorrent `buddy-cli` runtime. It shares the Llama TT runner
with `models/llama31_tt`, but uses completion-style prompts and a default cache
length of 128 tokens.

## Shell Variables

```bash
cd /path/to/buddy-mlir

export BUDDY_REPO_ROOT=$(pwd)
export BUDDY_BUILD="$BUDDY_REPO_ROOT/build-tenstorrent"
export TTMLIR_BUILD="$BUDDY_REPO_ROOT/build-ttmlir"
export TTMLIR_TOOLCHAIN_DIR="$BUDDY_REPO_ROOT/build-ttmlir-toolchain"
```

## Python Dependencies

Run this after the Tenstorrent environment from
[TenstorrentEnvironment.md](../../docs/TenstorrentEnvironment.md) is ready.

```bash
cd "$BUDDY_REPO_ROOT/thirdparty/tt-mlir"
source env/activate
cd "$BUDDY_REPO_ROOT"

"$TTMLIR_TOOLCHAIN_DIR/venv/bin/python" -m pip install --index-url https://download.pytorch.org/whl/cpu torch
"$TTMLIR_TOOLCHAIN_DIR/venv/bin/python" -m pip install \
  transformers accelerate safetensors sentencepiece
"$TTMLIR_TOOLCHAIN_DIR/venv/bin/python" -c "import torch, transformers; print('llama frontend deps ok')"
```

## Build

Build the canonical batch-1 package:

```bash
cmake -S "$BUDDY_REPO_ROOT" -B "$BUDDY_BUILD" \
  -DBUDDY_BUILD_LLAMA32_TT_MODEL=ON \
  -DBUDDY_LLAMA32_MODEL_PATH=/path/to/Llama-3.2-3B
cmake --build "$BUDDY_BUILD" --target buddy-cli llama32_tt_rax
```

The package is written to:

```text
$BUDDY_BUILD/models/llama32_tt/llama32_tt.rax
```

Build fixed-batch packages by setting `BUDDY_LLAMA32_FIXED_BATCH_SIZES`.
The default cache length is 128 tokens.

```bash
cmake -S "$BUDDY_REPO_ROOT" -B "$BUDDY_BUILD" \
  -DBUDDY_BUILD_LLAMA32_TT_MODEL=ON \
  -DBUDDY_LLAMA32_MODEL_PATH=/path/to/Llama-3.2-3B \
  -DBUDDY_LLAMA32_FIXED_BATCH_SIZES="32"
cmake --build "$BUDDY_BUILD" --target llama32_tt_b32_rax
```

This writes:

```text
$BUDDY_BUILD/models/llama32_tt_b32/llama32_tt_b32.rax
```

## Run

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

"$BUDDY_BUILD/bin/buddy-cli" \
  --model "$BUDDY_BUILD/models/llama32_tt/llama32_tt.rax" \
  --prompt "I like taking walks in the" \
  --max-tokens 32 \
  --temperature 0
```

Fixed-batch run:

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

- Llama 3.2 uses plain BOS + prompt tokenization by default. Pass
  `--chat-template` only when you want chat formatting.
- `--prompt-file` may contain either one prompt, which is repeated for the
  full batch, or exactly `--batch-size` prompts.
- Batch output prints user 0 by default. Use `--print-all-batch` or set
  `BUDDY_LLAMA31_PRINT_ALL_BATCH=1` to print every user.
- `--prompt-length 32` matches the reference Llama 3.2 generation setup.
- EOS/EOT stopping is enabled by default. Set `BUDDY_LLAMA31_IGNORE_EOS=1`
  before packaging only for fixed-length benchmark packages.
- Set `TT_LOGGER_LEVEL=FATAL`, `TT_METAL_LOGGER_LEVEL=FATAL`, and
  `TTMLIR_RUNTIME_LOGGER_LEVEL=FATAL` to suppress runtime logs.
