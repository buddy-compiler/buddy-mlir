# Qwen3 (Qwen/Qwen3-0.6B)

Causal language model
[Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)
served through the `buddy-cli` / `.rax` runtime. The build imports the PyTorch
model, compiles its dense Qwen3 decoder (28 layers, GQA, f32) to MLIR, links it
as a shared library, and packs a `qwen3.rax` manifest plus an `InferenceRunner`
plugin that `buddy-cli` loads at run time. It uses the shared `buddy_add_model`
entry with the default `llm_prefill_decode` model kind.

## Prerequisites

- A built LLVM/MLIR and `buddy-mlir` (see the top-level [README](../../README.md)).
- The Python environment that `buddy-mlir` was built against (the `buddy-env`
  venv: Python 3.12, `torch`, `transformers`, `safetensors`, `numpy`).
- The model weights. Two options:
  - A local HuggingFace snapshot of `Qwen/Qwen3-0.6B` (downloaded once via
    `huggingface-cli` or `transformers`); pass its directory with
    `--local-model` (or the `BUDDY_QWEN3_LOCAL_MODEL` CMake variable). This is
    required for an offline build.
  - Network access at build time: `import_model.py` falls back to downloading
    `Qwen/Qwen3-0.6B` from the HuggingFace Hub if no local model is given.

## Build

Configure + build from the repo root with the shared
`tools/buddy-codegen/build_model.py` entry point (same one used by DeepSeek R1):

```bash
python3 tools/buddy-codegen/build_model.py \
  --spec models/qwen3/specs/f32.json \
  --build-dir build \
  --local-model /path/to/Qwen3-0.6B-snapshot
```

`--local-model` maps to the `BUDDY_QWEN3_LOCAL_MODEL` cache variable, which the
`buddy_add_model` macro forwards to the generic importer as the
`DEEPSEEKR1_MODEL_PATH` environment variable.

Alternatively, configure a manual CMake build with the model gate enabled
(`buddy_add_model(NAME qwen3 …)` is wired through `models/CMakeLists.txt`):

```bash
cmake -S . -B build \
  -DBUDDY_BUILD_QWEN3_MODEL=ON \
  -DBUDDY_QWEN3_LOCAL_MODEL=/path/to/Qwen3-0.6B-snapshot
cmake --build build --target qwen3_rax
```

The import and the two MLIR compiles run as part of the target and are the slow
part. The build emits these artifacts under `build/models/qwen3/`:

| File | Description |
| --- | --- |
| `qwen3.rax` | Model manifest (`model_name`, runner library, vocab) |
| `qwen3_runner.so` | `InferenceRunner` plugin loaded by `buddy-cli` |
| `qwen3_model.so` | Compiled Qwen3 decoder kernels (flat-C ABI) |
| `arg0.data` | External weight blob (~596 M f32 elements) |
| `vocab.txt` | Tokenizer vocabulary (Qwen BPE) |
| `layer_partitioned/` | Layer-partitioned prefill MLIR (compile time) |

## Run

```bash
./build/bin/buddy-cli \
  --model ./build/models/qwen3/qwen3.rax \
  --prompt "Hello, my name is"
```

- `--prompt "<text>"` runs single-shot generation.
- `--max-new-tokens <N>` overrides the number of generated tokens.
- `--interactive --chat-template <path.json>` starts a REPL-style multi-turn
  session (requires a chat-template JSON, e.g. a Qwen3 `ChatML` template;
  `--prompt` then sets the system prompt).
- `--sampling-config` / sampler flags control decoding (greedy by default).

Stop tokens are `[151645]` (`<|im_end|>`), Qwen3's EOS token.

## Notes

- The `.rax` and other artifacts live in the **build** directory
  (`build/models/qwen3/`), not in the source tree. Use the `build/` prefix in
  the `--model` path.
- The model has `tie_word_embeddings = true`, so the embedding and `lm_head`
  share one weight tensor; the spec's `weights_override.total` (596,049,984) is
  the deduplicated element count written to `arg0.data` (includes the 64-element
  RoPE `inv_freq` buffer traced alongside the 310 unique weight tensors).
- KV cache is `56 × {1, 8, 1024, 128}` f32 (28 layers × K/V), matching
  `head_num=8`, `hidden_size=128`, `kv_layers=56` in the generated config.
- The weights and MLIR artifacts are large; keep them out of the source tree
  (see `.gitignore`).
