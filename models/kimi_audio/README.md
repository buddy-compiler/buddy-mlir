# Kimi-Audio-7B-Instruct

[moonshotai/Kimi-Audio-7B-Instruct](https://huggingface.co/moonshotai/Kimi-Audio-7B-Instruct),
an audio-text-to-text multimodal LLM, served through the `buddy-cli` / `.rax`
runtime. The build imports the PyTorch model, compiles a fixed-shape
single-forward graph to MLIR, links `kimi_audio_model.so`, and packs a
`kimi_audio.rax` manifest plus an `InferenceRunner` plugin that `buddy-cli`
loads at run time.

This integration is **text-only**: the compiled forward runs the whole decoder
over a padded text sequence with whisper features disabled. Audio tokens are
beyond the scope of this runner.

## Prerequisites

- A built LLVM/MLIR and `buddy-mlir` (see the top-level [README](../../README.md)),
  configured with `-DBUDDY_MLIR_ENABLE_PYTHON_PACKAGES=ON`.
- The Python environment that `buddy-mlir` was built against, with `torch`,
  `transformers`, and the Buddy Python frontend available. This is only needed
  at **build time** to import the PyTorch model; the packaged `.rax` has no
  Python dependency at run time.
- A local HuggingFace `moonshotai/Kimi-Audio-7B-Instruct` snapshot directory.
  The repository intentionally does not provide a default local path.
- **A large machine.** Import produces `arg0.data` with
  `params_size = 9,735,127,168` f32 elements (~36 GiB); running the kernel
  additionally holds the weights in RAM (tens of GB) plus the two
  `1 x 1024 x 168448` logit buffers.

## Build

`kimi_audio` is wired through `buddy_add_model(MODEL_KIND single_forward)` with
its own importer/manifest generator (`codegen/import-kimi_audio.py` /
`codegen/gen_kimi_audio_manifest.py`, the same pluggable
`IMPORT_SCRIPT`/`MANIFEST_SCRIPT` mechanism ColBERTv2 uses). Build it directly
with CMake:

```bash
cd buddy-mlir
conda activate buddy

cmake -G Ninja -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUDDY_MLIR_ENABLE_PYTHON_PACKAGES=ON \
  -DBUDDY_BUILD_KIMI_AUDIO_MODEL=ON \
  -DBUDDY_KIMI_AUDIO_MODEL_PATH=/path/to/Kimi-Audio-7B-Instruct
cmake --build build --target kimi_audio_rax
```

`-DBUDDY_KIMI_AUDIO_MODEL_PATH` must point to a local HuggingFace-format
Kimi-Audio snapshot directory (it is staged into the build as
`BUDDY_KIMI_AUDIO_MODEL_PATH` and forwarded to the importer via the
`KIMI_AUDIO_MODEL_PATH` environment variable). The importer patches the remote
code (`modeling_moonshot_kimia.py`) for CPU fullgraph tracing, then exports the
forward graph.

The build emits these artifacts under `build/models/kimi_audio/`:

| File | Description |
| --- | --- |
| `kimi_audio.rax` | Model manifest (embeds weights, `kimi_audio_model.so`, `vocab.txt`, runner) |
| `kimi_audio_model.so` | Compiled MLIR kernels (exports `_mlir_ciface_forward`) |
| `kimi_audio_runner.so` | `InferenceRunner` plugin loaded by `buddy-cli` |
| `arg0.data` | Flattened traced weights (~36 GiB) |
| `vocab.txt` | Qwen byte-level BPE vocabulary, staged from `examples/BuddyDeepSeekR1/` |

## Run

```bash
./build/bin/buddy-cli \
  --model ./build/models/kimi_audio/kimi_audio.rax \
  --prompt "Hello, world"
```

- `--prompt "<text>"` selects the text to generate logits for.
- `--no-stats` suppresses runner logs and prints only the argmax token ids.

The output is two JSON lists of `max_seq_len` integers: the argmax token ids of
`audio_logits` and of `text_logits`, one per padded position.

## Notes

- The `.rax` and other artifacts live in the **build** directory
  (`build/models/kimi_audio/`), not in the source tree. Use the `build/` prefix
  in the `--model` path.
- Tokenization is a pure C++ reimplementation of Qwen's byte-level BPE tokenizer
  (`buddy/LLM/TextContainer.h`) built into `KimiAudioRunner.cpp`, loaded from
  the staged `vocab.txt`. Kimi-Audio's tokenizer is Qwen2.5-based but the HF
  snapshot ships **no** tokenizer files, so the shared Qwen vocab staged by
  `buddy_add_model` is a documented approximation; token ids may differ from
  the official `Qwen2Tokenizer`.
- The MLIR forward ABI is
  `forward(weights, input_ids, position_ids) -> (audio_logits, text_logits)`
  with shapes `memref<9735127168xf32>`, `memref<1x1024xi64>`, `memref<1x1024xi64>`
  and `memref<1x1024x168448xf32>` outputs. `params_size` is the sum of all
  traced weights **excluding** the whisper `vq_adaptor` (dead in the text-only
  graph), so it is smaller than the raw safetensors total.
- `max_seq_len` is fixed by `models/kimi_audio/specs/f32.json`. Change the spec
  (or pass a different spec via `--spec`) to re-target a different sequence
  length; import time grows linearly with it.
- For a quick accuracy check, compare against
  `AutoModelForCausalLM.from_pretrained(<local snapshot>)` using the same
  all-zero `text_input_ids` / `is_continuous_mask`, all-ones `attention_mask`,
  explicit `position_ids`, `whisper_input_feature=None`, and `return_dict=False`.
