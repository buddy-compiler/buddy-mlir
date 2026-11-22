# MMS vocoder — facebook/mms-tts-eng

This package compiles the English MMS-TTS model's HiFi-GAN vocoder into a
CPU `.rax` package. The input is a float32 latent tensor with shape
`[1, 192, max_seq_len]`; the output is a waveform with
`max_seq_len * 256` samples at 16 kHz. The default specification uses
30 latent frames and produces 7,680 samples.

The compiled stage contains 14,327,424 parameters. Text encoding, duration
prediction and flow decoding run in PyTorch when preparing validation inputs;
they are not included in the `.rax` package. `--prompt` therefore takes a
latent file path, not a sentence.

## Build

Build LLVM/MLIR and Buddy with Python bindings as described in the root
[README](../../README.md). Use the repository's Python dependencies. Download
`facebook/mms-tts-eng` revision
`c71de0fe7204c83f1c10820a7d696d0b450048ba`, or supply an existing snapshot
containing `config.json`, model weights and tokenizer files.

From the repository root, with the same Python environment used to build
Buddy:

```sh
python tools/buddy-codegen/build_model.py \
  --spec models/mms/specs/f32.json \
  --build-dir build \
  --local-model /path/to/mms-tts-eng
cmake --build build --target buddy-cli
```

For Clang 24, use matching `flatc` and FlatBuffers headers from version
24.3.25. If they are installed outside the system prefix, add
`--cmake-args=-DFLATC_EXECUTABLE=/path/to/flatbuffers/bin/flatc` and
`--cmake-args=-DFLATBUFFERS_INCLUDE_DIR=/path/to/flatbuffers/include` to the
build command. The old system headers may fail to compile with this compiler.

Omitting `--local-model` downloads the revision specified in the model spec.
The build imports one complete vocoder graph, compiles it, and packages
`mms_model.so`, `mms_runner.so`, `mms_server.so`, weights and runtime dependencies in
`build/models/mms/mms.rax`. The package can be moved without its build directory.

For an equivalent CMake invocation after configuring the Buddy build:

```sh
cmake -S . -B build \
  -DBUDDY_MLIR_ENABLE_PYTHON_PACKAGES=ON \
  -DBUDDY_BUILD_MMS_MODEL=ON \
  -DBUDDY_MMS_SPEC="$PWD/models/mms/specs/f32.json" \
  -DBUDDY_MMS_MODEL_PATH=/path/to/mms-tts-eng
cmake --build build --target mms_rax buddy-cli
```

## Validate and run

Run the accuracy check with the same snapshot used for compilation:

```sh
OMP_NUM_THREADS=4 python models/mms/validate_accuracy.py \
  --local-model /path/to/mms-tts-eng \
  --model-dir build/models/mms \
  --buddy-cli build/bin/buddy-cli \
  --output-dir build/mms-validation
```

The check uses three transcripts from the
[LibriSpeech test-clean split](https://www.openslr.org/12/), recorded with
their utterance IDs and dataset source in `validation_texts.json`. It
synthesizes them with the pretrained VITS model, captures the vocoder inputs,
and tests windows from the beginning,
middle and end of each latent sequence. It compares every compiled waveform
sample against PyTorch (`atol=1e-4`, `rtol=1e-3`), runs the relocated `.rax`
through `buddy-cli`, and rejects truncated, oversized and non-finite inputs
as well as truncated weights. It also checks malformed dimensions, tensor-size
overflow and inconsistent waveform metadata in packages built with `rax-pack`.
`accuracy.json` records the errors; paired `.reference.wav` and `.compiled.wav`
files contain the compared audio. Add repeated `--text` arguments to test other
English sentences.

The generated latent files can also be passed directly to the runner:

```sh
build/bin/buddy-cli --model build/models/mms/mms.rax \
  --prompt build/mms-validation/text0_frame0.f32 --no-stats
```

The runner prints the sample count, minimum, maximum, mean, RMS and first eight
samples as JSON. Input files must contain exactly `1 * 192 * max_seq_len`
little-endian float32 values in channel-first order. An empty prompt retains
the zero-latent diagnostic; it is not used for accuracy validation.

## Serve the vocoder

Build `buddy-server` from the same source tree as the model package. The
package's `model_library` selects the resident vocoder plugin; `runner_library`
continues to select the CLI runner. Both use the same loader and compiled
kernel. Weights are loaded once, and the server serializes calls to the plugin.

```sh
cmake --build build --target buddy-server mms_rax
build/bin/buddy-server --model build/models/mms/mms.rax \
  --host 127.0.0.1 --port 8080
```

`POST /v1/audio/vocoder` accepts a JSON object with `latents` and an optional
`model` string (`facebook/mms-tts-eng`). `latents` is a flat array in contiguous
`[1, flow_size, max_seq_len]` order, exactly 5,760 float32 values for the default
package. It can be prepared from a genuine validation window:

```sh
python - <<'PYTHON'
import json
import numpy as np
from pathlib import Path
latents = np.fromfile("build/mms-validation/text0_frame0.f32", dtype=np.float32)
Path("build/mms-request.json").write_text(json.dumps({"latents": latents.tolist()}))
PYTHON
curl --fail-with-body http://127.0.0.1:8080/v1/audio/vocoder \
  -H 'Content-Type: application/json' --data-binary @build/mms-request.json
```

The response contains `model`, `sample_rate` (16,000 Hz), `channels` (1),
`inference_ms`, and the complete `audio` array of float32 waveform samples.
The samples are not clipped or quantized. The endpoint is a vocoder extension,
not the OpenAI text-to-speech API: text, streaming, batch input, paths, and
unknown fields are rejected. Invalid shapes, non-numeric or non-finite
latents, and a mismatched model return HTTP 400. The HTTP parser accepts at
most 1,048,576 latent elements; the plugin additionally requires the package's
exact latent count. Existing service endpoints return an unsupported-operation
error when this model is loaded.

After running `validate_accuracy.py`, validate the production HTTP path:

```sh
python models/mms/validate_server.py \
  --buddy-server build/bin/buddy-server \
  --package build/models/mms/mms.rax \
  --validation-dir build/mms-validation \
  --output-dir build/mms-server-validation
```

This starts an isolated loopback server with a relocated package and no
`LD_LIBRARY_PATH`, compares all saved real VITS windows to their float32
PyTorch reference waveforms, and checks repeated and concurrent requests,
invalid inputs, unsupported endpoints, and recovery after errors. It records
inference and HTTP wall times; `--request-timeout` (default 300 seconds) includes
time waiting behind other serialized requests. The process
is stopped on success or failure. Run both validators again after rebuilding
with a different static frame length.

For direct C ABI lifecycle checks, convert one reference waveform produced by
`validate_accuracy.py` to a raw float32 file and run the production plugin:

```sh
cmake --build build --target mms-server-abi-test
python - <<'PYTHON'
import soundfile as sf
waveform, rate = sf.read("build/mms-validation/text0_frame0.reference.wav", dtype="float32")
assert rate == 16000
waveform.tofile("build/mms-reference.f32")
PYTHON
build/bin/mms-server-abi-test build/models/mms/mms.rax \
  build/mms-validation/text0_frame0.f32 build/mms-reference.f32
```

These checks cover unloaded access, malformed ABI requests, cancellation,
expired deadlines, failed callbacks, failed reloads and successful recovery,
including a full reference-waveform comparison after reloading.

The vocoder adds a service kind and operation to the existing C ABI; it does
not change existing enum values or the sizes of the request/event unions.
Older servers reject the new service kind. Rebuild the server when using a
package that includes `mms_server.so`.

## Import and runtime checks

The importer requires a single-speaker VITS checkpoint whose flow width,
upsampling rates and parameter count match the spec. It rejects fragmented
traces instead of compiling only the first graph. Transposed convolutions are
lowered into stride-one phases and interleaved to preserve the HiFi-GAN
upsampling operation. Their source tensors are found by SSA reference rather
than by instruction position, including when larger tensors require explicit
zero initialization.

The manifest carries the latent dimensions, upsampling factor, waveform size
and parameter count. The runner checks these values and the weight and latent
file sizes before calling the kernel. Shared-library handles and returned
waveform buffers are released on both success and exceptions.

To check another static input length, copy the spec, change `max_seq_len`, and
pass that spec to both `build_model.py --spec` and
`validate_accuracy.py --spec`. Rebuild the package before validating it. The
32-frame case exercises the frontend's transition to explicit convolution
accumulator initialization.
