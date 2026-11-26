# Buddy Compiler StableLM2 Example

## Introduction

This example shows how to use Buddy Compiler to compile a StableLM-2 1.6B model
to MLIR code then run it on a RISC-V (SpacemiT K3) device.

Model: `stabilityai/stablelm-2-1_6b` (24 layers, hidden 2048, 32 heads MHA,
head_dim 64, intermediate 5632, vocab 100352 rows / 100289 real tokens,
partial RoPE factor 0.25, tied embeddings off, f32 params 1,644,515,336).

## How to run on non-RISC-V device

0. Enter Python virtual environment.

Follow the buddy-mlir LLVM/MLIR and buddy-mlir build steps in the root README
(same as the Qwen3 example). The Python bindings must match the interpreter
used for the build (here: Python 3.14 venv, see `SETUP_PY314_REPORT.md`).

1. Set model environment variable.

```bash
$ export STABLELM2_1_6B_MODEL_PATH=/path-to-stablelm2-model/

// For example:
$ export STABLELM2_1_6B_MODEL_PATH=/home/xxx/stablelm-2-1_6b
```

Alternatively, leave the path blank and `import-stablelm2.py` will download
`stabilityai/stablelm-2-1_6b` from Hugging Face automatically.

2. Enable the example at configure time.

```bash
$ cd buddy-mlir/build
$ cmake -G Ninja .. -DBUDDY_STABLELM2_EXAMPLES=ON
```

3. Import (trace) the model: produces forward/subgraph MLIR + `arg0.data`.

```bash
$ cd buddy-mlir
$ .venv314/bin/python examples/BuddyStableLm2/import-stablelm2.py
# outputs land in build/examples/BuddyStableLm2/
# arg0_stablelm2_1_6b.data ~ 6.58 GB — keep it on disk, not in tmpfs
```

4. Build and run.

```bash
$ ninja buddy-stablelm2-1.6b-run
$ printf '%s\n' 'The capital of France is' | ./bin/buddy-stablelm2-1.6b-run
```

Expected (greedy, f32, HF-identical first 30 tokens; `generateLen = 29`):

```
 a city of 1.6 million people, and the second largest city in Europe.
 Paris is a city of art, culture, and history.
```

## Notes on this adaptation

- Tokenizer: bare byte-level BPE continuation (no chat template). Added as
  `tokenizeStableLm2` / `revertStableLm2` in `frontend/Interfaces/buddy/LLM/TextContainer.h`.
  `bos == eos == pad == 100257`; vocab.txt is generated from the HF tokenizer
  (`gen_vocab.py`, line number == token id, 100289 lines).
- Argmax is clamped to `RealVocabSize = 100289` (embedding rows 100289..100351
  are padding rows and must never be sampled).
- Prefill graph is exact vs HF (teacher-forced). The single-token decode graph
  currently mis-masks pad positions at runtime for this model family
  (verified by poisoning and per-layer KV diffs); this harness therefore
  re-runs the prefill graph per generated step (O(n^2), correctness-first
  reference path). Decode-graph fix is tracked upstream.
- ABI: prefill = 2 in / 49 out (48 KV memrefs + logits); decode = 74 in /
  73 out (cache_position_out, 23x(KV,KV,dummy), last layer has no dummy,
  logits). Struct field order must exactly match the MLIR return tuple.
