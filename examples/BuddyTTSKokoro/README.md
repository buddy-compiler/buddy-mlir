# BuddyTTSKokoro

This example imports and runs a fixed-shape Kokoro-82M TTS pipeline with
Buddy MLIR.

Kokoro predicts token durations before it builds the frame alignment consumed by
the vocoder.  That alignment width is data dependent, so this example follows a
two-stage static pipeline instead of forcing the whole model into one static
graph:

- `forward_predictor.mlir`, `subgraph0_predictor.mlir`
  - encode fixed phoneme tokens and reference style
  - predict token durations
  - return the intermediate tensors needed by the vocoder
- `forward_vocoder.mlir`, `subgraph0_vocoder.mlir`
  - consume a fixed frame-index tensor built by the C++ driver
  - synthesize the final waveform

The C++ driver computes the duration-to-frame bridge between those two static
graphs and writes a WAV file.

## Build

The sample follows the same opt-in structure as the other examples.  Enabling
`BUDDY_TTS_KOKORO_EXAMPLES` imports the MLIR/data files, compiles the generated
MLIR objects, links the `KOKORO` static library, and builds the native runner:

```bash
cmake -S . -B build -DBUDDY_TTS_KOKORO_EXAMPLES=ON
cmake --build build --target buddy-kokoro-run
```

The generated files are placed in `build/examples/BuddyTTSKokoro/`:

- `forward_predictor.mlir`, `subgraph0_predictor.mlir`
- `forward_vocoder.mlir`, `subgraph0_vocoder.mlir`
- `arg0_predictor.data`, `arg0_vocoder.data`
- `input_ids.data`, `ref_s.data`

The two `arg0_*.data` files contain float model weights packed for the generated
forward wrappers.  The predictor stage also has an int64 Buddy ABI input for
deterministic runtime state; the C++ driver recreates that state locally.

## Run

```bash
./build/bin/buddy-kokoro-run
```

The default output is:

```text
build/examples/BuddyTTSKokoro/kokoro_static.wav
```

The runner accepts explicit input/output paths:

```bash
./build/bin/buddy-kokoro-run \
  --params0 build/examples/BuddyTTSKokoro/arg0_predictor.data \
  --params1 build/examples/BuddyTTSKokoro/arg0_vocoder.data \
  --tokens build/examples/BuddyTTSKokoro/input_ids.data \
  --ref-s build/examples/BuddyTTSKokoro/ref_s.data \
  --output build/examples/BuddyTTSKokoro/kokoro_static.wav
```

## Input Text

The default fixed-shape sample uses:

- `input_ids` shape `[1, 16]`
- `ref_s` shape `[1, 256]`
- speed `1.0`
- phoneme string `həlˈoʊ wˈɜɹld`

`--phonemes` expects a Kokoro phoneme string, not raw English text.  Passing raw
English through `--phonemes` treats the characters as already-phonemized input
and can produce harsh or unnatural audio.

If local G2P assets are installed, the importer can phonemize raw text before
export:

```bash
python \
  examples/BuddyTTSKokoro/import-kokoro.py \
  --output-dir build/examples/BuddyTTSKokoro \
  --text "Hello world."
```

Without those local G2P assets, use `--phonemes` directly:

```bash
python \
  examples/BuddyTTSKokoro/import-kokoro.py \
  --output-dir build/examples/BuddyTTSKokoro \
  --phonemes "həlˈoʊ wˈɜɹld"
```

If the input changes enough to alter the predicted alignment length or generated
audio length, update the static constants in `buddy-kokoro-main.cpp` to match
the newly generated MLIR/data.
