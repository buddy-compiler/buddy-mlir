# E5-Mistral

E5-Mistral-7B-Instruct dense sentence embedding with
[intfloat/e5-mistral-7b-instruct](https://huggingface.co/intfloat/e5-mistral-7b-instruct),
served through the `buddy-cli` / `.rax` runtime. The build imports the PyTorch
Mistral-architecture **encoder** (`MistralModel`), compiles the fixed-shape
forward graph to MLIR, links `e5_mistral_model.so`, and packs an
`e5_mistral.rax` manifest plus an `InferenceRunner` plugin that `buddy-cli`
loads at run time.

The runner emits the 4096-dimensional sentence embedding taken from the last
token of the encoder output (`last_hidden_state[:, -1]`). For the e5-mistral
tokenizer the last position always holds the `</s>` end-of-sequence token, so
this matches the PyTorch reference `model(...).last_hidden_state[0, -1, :]`.

## Prerequisites

- A built LLVM/MLIR and `buddy-mlir` (see the top-level [README](../../README.md)),
  configured with `-DBUDDY_MLIR_ENABLE_PYTHON_PACKAGES=ON`.
- The Python environment that `buddy-mlir` was built against, with `torch`,
  `transformers`, and the Buddy Python frontend available. This is only needed
  at **build time** to import the PyTorch model; the packaged `.rax` has no
  Python dependency at run time (see Notes).
- A local HuggingFace `intfloat/e5-mistral-7b-instruct` snapshot directory
  (sharded `model-*.safetensors`/`pytorch_model-*.bin`). The repository does
  not provide a default local path.

## Build

`e5-mistral` is wired through `buddy_add_model(MODEL_KIND single_forward)` with
its own importer/manifest generator (`codegen/import-e5-mistral.py` /
`codegen/gen_e5_mistral_manifest.py`, the same pluggable
`IMPORT_SCRIPT`/`MANIFEST_SCRIPT` mechanism BGE-M3 and ColBERTv2 use). Like
ColBERTv2 it is a custom `single_forward` model, not an LLM prefill/decode
model, so it is gated behind a `BUDDY_BUILD_E5_MISTRAL_MODEL` option. Add the
standard gate to `models/CMakeLists.txt`:

```cmake
if(BUDDY_BUILD_E5_MISTRAL_MODEL)
  add_subdirectory(e5-mistral)
endif()
```

then configure and build:

```bash
cd buddy-mlir
conda activate buddy

cmake -G Ninja -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUDDY_MLIR_ENABLE_PYTHON_PACKAGES=ON \
  -DBUDDY_BUILD_E5_MISTRAL_MODEL=ON \
  -DBUDDY_E5_MISTRAL_MODEL_PATH=/path/to/e5-mistral-7b-instruct
cmake --build build --target e5_mistral_rax
```

`-DBUDDY_E5_MISTRAL_MODEL_PATH` must point to a local HuggingFace-format
`intfloat/e5-mistral-7b-instruct` snapshot directory (it is staged into the
build and forwarded to the importer via the `E5_MISTRAL_MODEL_PATH`
environment variable).

The build emits these artifacts under `build/models/e5-mistral/`:

| File | Description |
| --- | --- |
| `e5_mistral.rax` | Model manifest (embeds weights, `e5_mistral_model.so`, `tokenizer.json`, runner) |
| `e5_mistral_model.so` | Compiled MLIR kernels (exports `_mlir_ciface_forward`) |
| `e5_mistral_runner.so` | `InferenceRunner` plugin loaded by `buddy-cli` |
| `arg0.data` | Flattened encoder weights (~28 GB, 7.1B f32) |
| `tokenizer.json` | Llama BPE tokenizer, staged from the local snapshot |

## Run

```bash
./build/bin/buddy-cli \
  --model ./build/models/e5-mistral/e5_mistral.rax \
  --prompt "query: How long does it take to land on the moon?"
```

- `--prompt "<text>"` selects the text to encode. The e5-mistral-instruct
  convention is to prefix queries with `query:` and documents with `passage:`.
- `--no-stats` suppresses runner logs and prints only the embedding vector.

The output is a JSON-like list of `hidden_size` (4096) floats: the last-token
sentence embedding.

## Notes

- The `.rax` and other artifacts live in the **build** directory
  (`build/models/e5-mistral/`), not in the source tree. Use the `build/`
  prefix in the `--model` path.
- Tokenization is a pure C++ reimplementation of the Llama BPE tokenizer
  (`E5MistralTokenizer.h`), loaded directly from the staged `tokenizer.json`.
  There is no Python/`transformers` dependency at run time, so a `.rax` built
  once can be copied to another machine (same architecture) and run with just
  `buddy-cli` and that single file.
- The imported graph uses `max_seq_len = 128` and `hidden_size = 4096` from
  `models/e5-mistral/specs/f32.json`; the spec's `params_size` is the exact
  element count of the generated `arg0.data` (7,110,660,160 f32), including
  the 64-element rotary `inv_freq` buffer. Change the spec (or pass a different
  spec via `--spec`) to re-target a different fixed sequence length.
- The MLIR forward ABI is
  `forward(weights, input_ids, attention_mask) -> last_hidden_state` (a single
  result). `_mlir_ciface_forward` therefore takes
  `(MemRef<float,3>* result, MemRef<float,1>* weights, MemRef<int64_t,2>*
  input_ids, MemRef<int64_t,2>* attention_mask)` — see the comment at the top
  of `E5MistralRunner.cpp`.
- The tokenizer reproduces `AutoTokenizer(text, padding="max_length",
  truncation=True, max_length=max_seq_len)`. This model's tokenizer config
  only sets `add_eos_token=True` (leaving `add_bos_token` unset), so the Llama
  tokenizer default applies and `<s>` IS prepended: the sequence is
  `[</s>] * pad + [<s>] + bpe(text) + [</s>]`, **left-padded** with `</s>`
  (pad == eos). The content is right-aligned and the last token is always the
  `</s>` EOS token whose hidden state is emitted as the sentence embedding.
- For a quick accuracy check, compare the `buddy-cli` output against
  `AutoModel.from_pretrained(<local e5-mistral snapshot>)` using the same
  tokenizer settings: padding to `max_length = 128`, truncation enabled, and
  `last_hidden_state[0, -1, :]`. The outputs should match closely. (Embedding
  normalization, as applied by the sentence-transformers `1_Pooling` wrapper,
  is not part of the base `MistralModel` and is intentionally left out.)
