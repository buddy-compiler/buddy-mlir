# Llama 3.1 8B Tenstorrent buddy-cli Path

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
  -DBUDDY_BUILD_LLAMA31_TT_MODEL=ON \
  -DBUDDY_LLAMA31_MODEL_PATH=/path/to/Llama-3.1-8B-Instruct
cmake --build "$BUDDY_BUILD" --target buddy-cli llama31_tt_rax
```

Output:

```text
$BUDDY_BUILD/models/llama31_tt/llama31_tt.rax
```

## Build Fixed Batch

Build a fixed-batch package.

```bash
cmake -S "$BUDDY_REPO_ROOT" -B "$BUDDY_BUILD" \
  -DBUDDY_BUILD_LLAMA31_TT_MODEL=ON \
  -DBUDDY_LLAMA31_MODEL_PATH=/path/to/Llama-3.1-8B-Instruct \
  -DBUDDY_LLAMA31_FIXED_BATCH_SIZES="4"
cmake --build "$BUDDY_BUILD" --target llama31_tt_b4_rax
```

Output:

```text
$BUDDY_BUILD/models/llama31_tt_b4/llama31_tt_b4.rax
```

Use `-DBUDDY_LLAMA31_MAX_CACHE_LEN=512` or `256` to build a smaller-cache
fixed-batch package.

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
  --model "$BUDDY_BUILD/models/llama31_tt/llama31_tt.rax" \
  --prompt "Hello" \
  --max-tokens 32 \
  --repeat-penalty 1.12 \
  --repeat-last-n 128
```

## Run with buddy-server

The same canonical package can be served through the resident plugin. Enable
the model target while configuring the project; the package target builds
`llama31_tt_server.so` and records it in the manifest as `model_library`.

```bash
conda run -n buddy-mlir cmake --build "$BUDDY_BUILD" \
  --target buddy-server llama31_tt_rax
source "$BUDDY_REPO_ROOT/thirdparty/tt-mlir/env/activate"
export TT_METAL_RUNTIME_ROOT="$BUDDY_REPO_ROOT/thirdparty/tt-mlir/third_party/tt-metal/src/tt-metal"
export TT_METAL_HOME="$TT_METAL_RUNTIME_ROOT"
"$BUDDY_BUILD/bin/buddy-server" \
  --model "$BUDDY_BUILD/models/llama31_tt/llama31_tt.rax" \
  --host 127.0.0.1 --port 8080
```

`POST /completion`, `POST /v1/chat/completions`, `POST /tokenize`, and
`POST /detokenize` use the package's tokenizer and chat-template resources.
Compatible serving requires a tiktoken-format `tokenizer.model`; loading
prefers `original/tokenizer.model` before `tokenizer.model`. The legacy
`tokenizer.json` fallback does not declare compatible raw codec or generation
capabilities. Ordinary spans use whole-span byte BPE without canonical
Llama/tiktoken regex pre-tokenization, so exact canonical token IDs are not
certified. See the [TT serving guide](../../tools/buddy-server/compatibility/llama31-tt.md)
for raw-byte decoding, supported modes, and tokenizer limits.

HTTP generation is greedy only. Omit `temperature`, `top_k`, `top_p`, `min_p`,
`repeat_penalty`, `repeat_last_n`, and `seed`; explicit non-null sampling fields
are rejected even at their default values, including `temperature:0`. The
TTNN backend declares compatible generation only when both prefill and decode
output descriptors support Float32/BFloat16 logits or Int32/Int64/UInt32 device
argmax IDs. Float16 packages can still load and generate through the legacy
path but do not declare compatible generation. Budget aliases use
`n_predict > max_completion_tokens > max_tokens`: 0 evaluates the prompt
without selecting tokens, -1 runs until stop/context, and positive values
count new tokens. Context overflow is rejected without truncation.

Chat preserves supported roles and multiple turns, renders its default or
configured JSON template once, and rejects unsupported roles/images. Request,
package, and template stops are combined. Requests are serialized over one
device/context; request-scoped state is reset on success, cancellation, or
exception. Only `batch_size=1` is accepted; `llama31_tt_bN.rax` is rejected by
the server. Both external (manifest-only) and embedded payload packages are
supported.

The production plugin creates a TTNN session while the model is loading. The
session keeps the prefill/decode Binary objects, 1x1 MeshDevice, static weight
tensors, tokenizer artifacts, and program descriptors resident; prompt inputs,
KV tensors, and cache-position inputs are request-scoped and released on
success, cancellation, or exception. A healthy server reports
`backend: "ttnn"` and does not require the fake environment variable.

```bash
curl -sS http://127.0.0.1:8080/health | jq .

curl -sS http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"llama31_tt","messages":[{"role":"user","content":"Explain MLIR in one sentence."}],"max_tokens":32}' | jq .

curl -N --no-buffer http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Write one sentence about Tenstorrent."}],"max_tokens":32,"stream":true}'

curl -sS http://127.0.0.1:8080/completion \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nHello<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n","max_tokens":32}' | jq .

curl -sS http://127.0.0.1:8080/tokenize \
  -H 'Content-Type: application/json' \
  -d '{"content":"Hello buddy-server","add_special":true,"count_only":false}' | jq .

curl -sS http://127.0.0.1:8080/detokenize \
  -H 'Content-Type: application/json' \
  -d '{"tokens":[128000,128001]}' | jq .
```

`llama31_tt_server.so` resolves TTNN runtime libraries through its build/install
RPATH. When running from a non-installed build, keep the tt-metal runtime and
the conda `buddy-mlir` Python library on `LD_LIBRARY_PATH`; a missing Binary,
artifact, tokenizer, or device is a load error and `/health` remains in the
`error` state. Default execution/resident tests inject a scripted serving token
source; portable plugin checks compile with `BUDDY_LLAMA31_TT_HAS_TTNN=0`.
The legacy `BUDDY_LLAMA31_TT_FAKE_EXECUTION=1` mode does not provide compatible
HTTP generation. These checks exercise the portable serving path without the
TTNN SDK or a device. Compile the SDK-enabled production targets and run the
model on a device to validate those execution paths.

## Run Fixed Batch

Run a fixed-batch package.

```bash
cat > /tmp/$USER/llama31_b4_prompts.txt <<'EOF'
Hello
What is machine learning?
Write one sentence about compiler optimization.
I like taking walks in the
EOF

"$BUDDY_BUILD/bin/buddy-cli" \
  --model "$BUDDY_BUILD/models/llama31_tt_b4/llama31_tt_b4.rax" \
  --prompt-file /tmp/$USER/llama31_b4_prompts.txt \
  --prompt-length 32 \
  --batch-size 4 \
  --max-tokens 64 \
  --print-all-batch \
  --temperature 0
```

## Basic Settings

- Llama 3.1 uses the default chat prompt. Pass `--chat-template` to override it.
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
- Generation cannot exceed the package cache length. For the default package,
  the cache length is 1024 tokens.

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

- Llama 3.1 uses chat formatting by default.
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
