# MoLFormer

MoLFormer, the chemistry Transformer encoder
[ibm/MoLFormer-XL-both-10pct](https://huggingface.co/ibm/MoLFormer-XL-both-10pct),
served through the `buddy-cli` / `.rax` runtime. The build imports the PyTorch
model, compiles the fixed-shape forward graph to MLIR, links
`molformer_model.so`, and packs a `molformer.rax` manifest plus an
`InferenceRunner` plugin that `buddy-cli` loads at run time.

The runner tokenizes a SMILES string and emits the pooled 768-dimensional
molecular embedding (masked average of the per-token hidden states) together
with the per-token hidden states from the last encoder layer.

## Prerequisites

- A built LLVM/MLIR and `buddy-mlir` (see the top-level [README](../../README.md)),
  configured with `-DBUDDY_MLIR_ENABLE_PYTHON_PACKAGES=ON`.
- The Python environment that `buddy-mlir` was built against, with `torch`,
  `transformers`, and the Buddy Python frontend available. This is only needed
  at **build time** to import the PyTorch model; the packaged `.rax` has no
  Python dependency at run time (see Notes).
- A local HuggingFace `ibm/MoLFormer-XL-both-10pct` snapshot directory.

## Build

`molformer` is wired through `buddy_add_model(MODEL_KIND single_forward)` with
its own importer/manifest generator (`codegen/import-molformer.py` /
`codegen/gen_molformer_manifest.py`). It is gated behind the
`BUDDY_BUILD_MOLFORMER_MODEL` CMake option:

```bash
cd buddy-mlir
source buddy-env/bin/activate

cmake -G Ninja -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUDDY_MLIR_ENABLE_PYTHON_PACKAGES=ON \
  -DBUDDY_BUILD_MOLFORMER_MODEL=ON \
  -DBUDDY_MOLFORMER_MODEL_PATH=/path/to/MoLFormer-XL-both-10pct
cmake --build build --target molformer_rax
```

`-DBUDDY_MOLFORMER_MODEL_PATH` must point to a local HuggingFace-format
MoLFormer snapshot directory (it is staged into the build and forwarded to the
importer via the `MOLFORMER_MODEL_PATH` environment variable).

The build emits these artifacts under `build/models/molformer/`:

| File | Description |
| --- | --- |
| `molformer.rax` | Model manifest (embeds weights, `molformer_model.so`, `vocab.txt`, runner) |
| `molformer_model.so` | Compiled MLIR kernels (exports `_mlir_ciface_forward`) |
| `molformer_runner.so` | `InferenceRunner` plugin loaded by `buddy-cli` |
| `arg0.data` | Flattened encoder weights (44,709,888 f32 elements) |
| `vocab.txt` | WordLevel SMILES vocabulary (2362 tokens) |

## Run

```bash
./build/bin/buddy-cli \
  --model ./build/models/molformer/molformer.rax \
  --prompt "CC(=O)Oc1ccccc1"
```

- `--prompt "<SMILES>"` selects the molecule to embed.
- `--no-stats` suppresses runner logs and prints only the pooled embedding.

The output is the 768-dimensional pooled molecular embedding vector.

## Traceability notes

MoLFormer's linear attention uses Generalized Random Fourier Features. With the
checkpoint's default `deterministic_eval=False`, every forward re-rolls the
random projection (`torch.randn` / `linalg.qr`), which cannot be AOT-traced into
a single static graph. `codegen/import-molformer.py` therefore:

1. forces every `MolformerFeatureMap` into deterministic mode (the projection
   weights are a persistent buffer, so they are part of the exported weights);
2. replaces `MolformerSelfAttention.forward` with an equivalent one that skips
   the `torch.equal(attention_mask, ...)` shape check, which would otherwise
   graph-break inside every attention layer.

The result is the **full 12-layer encoder as a single graph** (44.7M params).
The model returns the pooler output too, so the AOT forward returns both the
per-token `last_hidden_state` and the pooled embedding.

## Notes

- The `.rax` and other artifacts live in the **build** directory
  (`build/models/molformer/`), not in the source tree.
- The tokenizer used by the runner is a C++ reimplementation of the checkpoint's
  WordLevel SMILES tokenizer over the staged `vocab.txt` (extracted from
  `tokenizer.json` at codegen time). Sequences are wrapped in `<bos>` / `<eos>`
  and padded to `max_seq_len` with `<pad>`.
