# BGE-M3

Dense text embedding with [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3),
served through the `buddy-cli` / `.rax` runtime. The build imports the PyTorch
encoder, compiles the fixed-shape forward graph to MLIR, links
`bge_m3_model.so`, and packs a `bge_m3.rax` manifest plus an `InferenceRunner`
plugin that `buddy-cli` loads at run time.

The initial integration targets dense embeddings only. The runner takes the CLS
vector from the encoder output and applies L2 normalization before printing the
1024-dimensional embedding.

## Prerequisites

- A built LLVM/MLIR and `buddy-mlir` (see the top-level [README](../../README.md)),
  configured with `-DBUDDY_MLIR_ENABLE_PYTHON_PACKAGES=ON`.
- The Python environment that `buddy-mlir` was built against, with `torch`,
  `transformers`, `sentencepiece`, and the Buddy Python frontend available.
- A local HuggingFace `bge-m3` snapshot directory. The repository intentionally
  does not provide a default local path.

## Build

Use the same `tools/buddy-codegen/build_model.py` entry point as the other
packaged models. It reads `model_family = "bge_m3"` from the spec, configures
CMake through `buddy_add_model(MODEL_KIND single_forward)`, imports the model
through the shared `import_model.py` dispatcher, compiles and links the kernels
through `compile_pipeline.py`, builds the runner plugin, and stages
`bge_m3.rax`:

```bash
cd buddy-mlir
conda activate buddy

python3 tools/buddy-codegen/build_model.py \
  --spec models/bge_m3/specs/base.json \
  --build-dir build \
  --local-model /path/to/bge-m3
```

`--local-model` must point to a local HuggingFace-format BGE-M3 snapshot. It is
forwarded to CMake as `BUDDY_BGE_M3_MODEL_PATH`. For lower-level iteration, the
CMake target is `bge_m3_rax`, but the Python entry above is the expected user
path.

The build emits these artifacts under `build/models/bge_m3/`:

| File | Description |
| --- | --- |
| `bge_m3.rax` | Model manifest (embeds weights, `bge_m3_model.so`, tokenizer assets, runner) |
| `bge_m3_model.so` | Compiled MLIR kernels (exports `_mlir_ciface_forward`) |
| `bge_m3_runner.so` | `InferenceRunner` plugin loaded by `buddy-cli` |
| `arg0.data` | Flattened encoder weights |
| `tokenizer.json` / `sentencepiece.bpe.model` | Tokenizer assets staged from the local snapshot |

## Run

```bash
./build/bin/buddy-cli \
  --model ./build/models/bge_m3/bge_m3.rax \
  --prompt "hello world"
```

- `--prompt "<text>"` selects the text to embed.
- `--no-stats` suppresses runner logs and prints only the embedding vector.

The output is a JSON-like dense embedding vector with dimension `1024`.

## Notes

- The `.rax` and other artifacts live in the **build** directory
  (`build/models/bge_m3/`), not in the source tree. Use the `build/` prefix in
  the `--model` path.
- Tokenization is delegated to the packaged Python helper
  `bge_m3_tokenize.py`, which uses the staged HuggingFace tokenizer assets. A
  pure-C++ tokenizer path is future work.
- The imported graph uses `max_seq_len = 512` from
  `models/bge_m3/specs/base.json`. Change the spec or pass a different spec via
  `--spec` for a different fixed sequence length.
- The generated MLIR forward ABI includes an internal token-type buffer. The
  runner fills it with zeros; it is not a monotonically increasing `position_ids`
  buffer.
- For a quick accuracy check, compare the `buddy-cli` output against
  `AutoModel.from_pretrained(<local BGE-M3 snapshot>)` using the same tokenizer
  settings: padding to `max_length`, truncation enabled, and `max_length = 512`.
  The dense embedding should match the normalized HF CLS vector closely.
