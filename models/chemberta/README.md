# ChemBERTa

ChemBERTa masked-LM encoding with
[DeepChem/ChemBERTa-77M-MLM](https://huggingface.co/DeepChem/ChemBERTa-77M-MLM),
a small RoBERTa-style chemistry encoder, served through the `buddy-cli` /
`.rax` runtime. The build imports the PyTorch `RobertaForMaskedLM` encoder,
compiles the fixed-shape forward graph to MLIR, links `chemberta_model.so`,
and packs a `chemberta.rax` manifest plus an `InferenceRunner` plugin that
`buddy-cli` loads at run time.

The runner emits the per-token masked-LM logits produced by the model's LM
head. For a 128-token sequence this yields 128 vectors of `vocab_size` (600)
logits; the last-position vector (`logits[0, -1, :]`) is the MLM prediction
used for accuracy checks.

## Prerequisites

- A built LLVM/MLIR and `buddy-mlir` (see the top-level [README](../../README.md)),
  configured with `-DBUDDY_MLIR_ENABLE_PYTHON_PACKAGES=ON`.
- The Python environment that `buddy-mlir` was built against, with `torch`,
  `transformers`, and the Buddy Python frontend available. This is only needed
  at **build time** to import the PyTorch model; the packaged `.rax` has no
  Python dependency at run time (see Notes).
- A local HuggingFace `DeepChem/ChemBERTa-77M-MLM` snapshot directory. The
  repository intentionally does not provide a default local path.

## Build

`chemberta` is wired through `buddy_add_model(MODEL_KIND single_forward)` with
its own importer/manifest generator (`codegen/import-chemberta.py` /
`codegen/gen_chemberta_manifest.py`, the same pluggable
`IMPORT_SCRIPT`/`MANIFEST_SCRIPT` mechanism ColBERTv2, BGE-M3 and Whisper use).
It is gated behind the `BUDDY_BUILD_CHEMBERTA_MODEL` CMake option, so build it
directly with CMake rather than `tools/buddy-codegen/build_model.py` (whose
`model_family` whitelist does not include `chemberta`):

```bash
cd buddy-mlir
conda activate buddy

cmake -G Ninja -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUDDY_MLIR_ENABLE_PYTHON_PACKAGES=ON \
  -DBUDDY_BUILD_CHEMBERTA_MODEL=ON \
  -DBUDDY_CHEMBERTA_MODEL_PATH=/path/to/ChemBERTa-77M-MLM
cmake --build build --target chemberta_rax
```

`-DBUDDY_CHEMBERTA_MODEL_PATH` must point to a local HuggingFace-format
ChemBERTa snapshot directory (it is staged into the build as
`BUDDY_CHEMBERTA_MODEL_PATH` and forwarded to the importer via the
`CHEMBERTA_MODEL_PATH` environment variable).

The build emits these artifacts under `build/models/chemberta/`:

| File | Description |
| --- | --- |
| `chemberta.rax` | Model manifest (embeds weights, `chemberta_model.so`, `vocab.txt`, runner) |
| `chemberta_model.so` | Compiled MLIR kernels (exports `_mlir_ciface_forward`) |
| `chemberta_runner.so` | `InferenceRunner` plugin loaded by `buddy-cli` |
| `arg0.data` | Flattened encoder weights |
| `vocab.txt` | Model vocabulary, staged from the local snapshot |

## Run

```bash
./build/bin/buddy-cli \
  --model ./build/models/chemberta/chemberta.rax \
  --prompt "c1ccncc1"
```

- `--prompt "<text>"` selects the SMILES/text to encode.
- `--no-stats` suppresses runner logs and prints only the logits.

The output is a JSON-like list of `max_seq_len` vectors of `vocab_size` logits,
one per token position (padded positions still emit a vector).

## Notes

- The `.rax` and other artifacts live in the **build** directory
  (`build/models/chemberta/`), not in the source tree. Use the `build/` prefix
  in the `--model` path.
- Tokenization is a pure C++ WordPiece reimplementation built into
  `ChembertaRunner.cpp`, loaded directly from the staged `vocab.txt`. There is
  no Python/`transformers` dependency at run time, so a `.rax` built once can be
  copied to another machine (same architecture) and run with just `buddy-cli`
  and that single file — no source tree, no Python environment. **Note:** the
  official ChemBERTa tokenizer is RoBERTa BPE (shipped as `vocab.json` +
  `merges.txt`); the snapshot must therefore contain a plain `vocab.txt` (the
  model vocabulary, one token per line) for the runner's built-in tokenizer.
  Special-token ids default to the RoBERTa ids (`<s>`=0, `</s>`=2, `<pad>`=1,
  `<unk>`=3) when those strings are absent from the file.
- The imported graph uses `max_seq_len = 128` and
  `position_buffer_size = 515` from `models/chemberta/specs/f32.json`. Change
  the spec (or pass a different spec via `--spec`) to re-target a different
  fixed sequence length.
- The MLIR forward ABI is
  `forward(weights, position_ids, input_ids, attention_mask) ->
  (logits)`. The runner fills `position_ids` with the sequence `0..514` and
  passes `<s>`/`</s>`-padded `input_ids` / `attention_mask`.
- For a quick accuracy check, compare the `buddy-cli` output against
  `AutoModelForMaskedLM.from_pretrained(<local ChemBERTa snapshot>)` using the
  same tokenizer settings: padding to `max_length = 128`, truncation enabled,
  and `return_dict = False`. The per-token logits should match closely (see the
  original PR's `validate_accuracy.py`, which compares `logits[0, -1, :]`).
