# Mistral 7B Instruct

Causal language generation with
[mistralai/Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2),
served through the `buddy-cli` / `.rax` runtime. The build imports the PyTorch
decoder, compiles the prefill + decode graphs to MLIR, links `mistral_model.so`,
and packs a `mistral.rax` manifest plus an `InferenceRunner` plugin that
`buddy-cli` loads at run time.

Mistral-7B uses a Llama-style BPE tokenizer, so the runner tokenizes with
`Text::tokenizeLlama` / `Text::revertLlama` and stops on the `</s>` EOS token
(id `2`).

## Prerequisites

- A built LLVM/MLIR and `buddy-mlir` (see the top-level [README](../../README.md)),
  configured with `-DBUDDY_MLIR_ENABLE_PYTHON_PACKAGES=ON`.
- The Python environment that `buddy-mlir` was built against, with `torch`,
  `transformers`, and the Buddy Python frontend available. This is only needed
  at **build time** to import the PyTorch model; the packaged `.rax` has no
  Python dependency at run time.
- A local HuggingFace `mistralai/Mistral-7B-Instruct-v0.2` snapshot directory
  (config.json + tokenizer files + model safetensors). The repository
  intentionally does not provide a default local path.

## Build

`mistral` is wired through `buddy_add_model` (the `llm_prefill_decode` model
kind) with the shared `tools/buddy-codegen` importer. It is gated behind the
`BUDDY_BUILD_MISTRAL_MODEL` CMake option, so build it directly with CMake:

```bash
cd buddy-mlir
conda activate buddy

cmake -G Ninja -S . -B build \
  -DBUDDY_MLIR_ENABLE_PYTHON_PACKAGES=ON \
  -DBUDDY_BUILD_MISTRAL_MODEL=ON \
  -DBUDDY_MISTRAL_LOCAL_MODEL=/path/to/Mistral-7B-Instruct-v0.2 \
  -DBUDDY_MISTRAL_COMPILE_JOBS=48
cmake --build build --target mistral_rax
```

Notes:

- Mistral-7B is a **7B model**: the f32 weight import writes a ~29 GB `arg0.data`
  and the two MLIR compiles (prefill + decode) take a while. Expect 20-60 minutes
  for the import, and give the build plenty of RAM and CPU.
- `BUDDY_MISTRAL_LOCAL_MODEL` points the importer at the local HF snapshot
  (`DEEPSEEKR1_MODEL_PATH`). If omitted, the importer falls back to the
  `hf_model_path` in the spec.
- The generated `.rax` (with the model `.so`, `arg0.data`, and `vocab.txt`)
  lands under `build/models/mistral/`.

## Run

Single-shot generation with the `.rax` manifest:

```bash
buddy-cli \
  --model build/models/mistral/mistral.rax \
  --prompt "What is the capital of France?" \
  --max-tokens 128 \
  --cpus 0-47
```

Interactive REPL (multi-turn) mode requires a Mistral chat template JSON:

```bash
buddy-cli \
  --model build/models/mistral/mistral.rax \
  --chat-template /path/to/mistral-instruct-chat-template.json \
  --interactive \
  --cpus 0-47
```

Sampler options (`--temperature`, `--top-k`, `--top-p`, `--seed`, ...) and
NUMA / CPU affinity flags behave as in the other `buddy-cli` models.

## Model Notes

- Architecture: 32-layer Llama-style decoder, hidden size 4096, 32 Q heads,
  8 KV heads (head dim 128), vocab 32000, RoPE theta 1e6, sliding window
  attention.
- Compiled shapes (from `specs/f32.json` via `gen_config.py`): KV cache
  `64 x {1, 8, 1024, 128}` f32 (`kv_layers = 2 * num_hidden_layers`,
  `head_num = num_key_value_heads`, `hidden_size = head_dim`,
  `max_token_len = 1024`).
- Parameter count (f32): 7,241,732,096 elements ≈ 29 GB of weights.
