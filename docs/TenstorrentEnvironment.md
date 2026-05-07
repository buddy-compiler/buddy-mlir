# Tenstorrent Environment

This page describes the optional environment used by Buddy's
`buddy-cli + llama31_tt + TTNN` path.

Normal Buddy builds do not need this. Enable it only on Tenstorrent machines
such as P150A, or when compiling TTIR to TTNN flatbuffers.

## Components

Buddy uses the following Tenstorrent components:

- `tt-mlir`: compiler tools such as `ttmlir-opt` and `ttmlir-translate`.
- `tt-metal` / TTNN runtime: fetched and configured by the `tt-mlir` build.
- `ttrt`: Python runtime package used to execute `.ttnn` flatbuffers.

The optional submodule is:

```bash
git submodule update --init --depth 1 thirdparty/tt-mlir
```

## Build tt-mlir

Follow the upstream instructions in
`thirdparty/tt-mlir/docs/src/getting-started.md`. The commands below are the
known-good shape used for the Buddy `llama31_tt` P150A smoke test.

Create or activate a Python environment first:

```bash
conda create -n buddy-ttmlir-p150a python=3.12 -y
conda activate buddy-ttmlir-p150a

python -m pip install --upgrade pip
python -m pip install cmake ninja
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch
python -m pip install \
  -r thirdparty/tt-mlir/env/build-requirements.txt \
  -r thirdparty/tt-mlir/env/ttnn-requirements.txt \
  -r thirdparty/tt-mlir/test/python/requirements.txt \
  -r thirdparty/tt-mlir/tools/ttrt/requirements.txt \
  transformers accelerate safetensors sentencepiece
```

Build the upstream tt-mlir toolchain:

```bash
export TTMLIR_TOOLCHAIN_DIR=/tmp/buddy-ttmlir-toolchain

cmake -G Ninja -S thirdparty/tt-mlir/env -B /tmp/buddy-ttmlir-env-build \
  -DCMAKE_C_COMPILER=/usr/bin/clang-20 \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++-20 \
  -DLLVM_BUILD_TYPE=MinSizeRel

cmake --build /tmp/buddy-ttmlir-env-build -j4
```

Build tt-mlir runtime, compiler tools, Python bindings, and `ttrt`:

```bash
cd thirdparty/tt-mlir
source env/activate

cmake -G Ninja -S . -B /tmp/buddy-ttmlir-build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/usr/bin/clang-20 \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++-20 \
  -DTTMLIR_ENABLE_RUNTIME=ON \
  -DTT_RUNTIME_ENABLE_PERF_TRACE=ON \
  -DTTMLIR_ENABLE_TESTS=OFF \
  -DTTMLIR_ENABLE_EXPLORER=OFF \
  -DTTMLIR_ENABLE_ALCHEMIST=OFF \
  -DCMAKE_CXX_FLAGS=-Wno-error=deprecated-declarations

cmake --build /tmp/buddy-ttmlir-build \
  --target ttmlir-opt ttmlir-translate TTMLIRPythonModules TTRTPythonModules ttrt \
  -j4

python -m pip install --force-reinstall --no-deps \
  /tmp/buddy-ttmlir-build/tools/ttrt/build/ttrt-*.whl
```

The exact compiler/Python requirements are controlled by upstream `tt-mlir`.
If the local docs differ from this summary, treat the upstream docs as the
source of truth.

Two practical notes from the current upstream `main` path:

- If a shallow `tt-mlir` checkout fails CMake version detection with
  `No tags can describe <commit>`, use a tagged tt-mlir revision, unshallow the
  submodule, or create a local test tag on the checked-out commit:
  `git -C thirdparty/tt-mlir tag -f v0.1 HEAD`.
- `TTMLIRPythonModules` and `TTRTPythonModules` are required by the Llama
  lowering and `python -m ttrt` paths. Building only `ttrt` can leave Python
  source modules out of the wheel on some upstream commits.

## Configure Buddy

After `tt-mlir` is built and its environment is active:

```bash
source thirdparty/tt-mlir/env/activate

cmake -S . -B /tmp/buddy-build-tt-p150a \
  -DMLIR_DIR=$TTMLIR_TOOLCHAIN_DIR/lib/cmake/mlir \
  -DLLVM_DIR=$TTMLIR_TOOLCHAIN_DIR/lib/cmake/llvm \
  -DPython3_EXECUTABLE=$TTMLIR_TOOLCHAIN_DIR/venv/bin/python \
  -DBUDDY_BUILD_LLAMA31_TT_MODEL=ON \
  -DBUDDY_ENABLE_TENSTORRENT=ON \
  -DBUDDY_TT_MLIR_SOURCE_DIR=$PWD/thirdparty/tt-mlir \
  -DBUDDY_TT_MLIR_BUILD_DIR=/tmp/buddy-ttmlir-build

cmake --build /tmp/buddy-build-tt-p150a --target buddy-cli rax-pack
```

When `TTMLIR_TOOLCHAIN_DIR` is set, Buddy prefers that toolchain's `flatc` and
FlatBuffers headers for RAX generation. This avoids mismatches between old
system `flatc` and newer toolchain headers.

The Llama 3.1 example also needs the Python LLM stack used by its frontend
capture scripts:

```bash
python -m pip install transformers accelerate safetensors sentencepiece
```

If your deployment uses a conda environment instead of the upstream
`tt-mlir/env/activate` venv, set it before sourcing the example helper:

```bash
export BUDDY_TT_CONDA_ENV=buddy-ttmlir-p150a
export BUDDY_BUILD=/tmp/buddy-build-tt-p150a
export TTMLIR_SOURCE=$PWD/thirdparty/tt-mlir
export TTMLIR_BUILD=/tmp/buddy-ttmlir-build
source models/llama31_tt/_env.sh
```

`BUDDY_ENABLE_TENSTORRENT=ON` checks that:

- `thirdparty/tt-mlir` exists,
- `ttmlir-opt` is discoverable,
- `ttmlir-translate` is discoverable,
- Python can import `ttrt` and `ttrt.runtime`.

Smoke-test the runtime before running a model:

```bash
python -c "import ttrt, ttrt.runtime, ttmlir.ir; print('tt runtime ok')"
python -m ttrt query
```

## Package TTNN Artifacts for buddy-cli

Given prebuilt Llama 3.1 TTNN flatbuffers and chat artifacts:

```bash
python3 tools/buddy-codegen/gen_tenstorrent_manifest.py \
  --prefill-ttnn models/llama31_tt/ttir_out_static/llama31_prefill_static_argattrs.ttnn \
  --decode-ttnn models/llama31_tt/ttir_out_static/llama31_decode_static_argattrs.ttnn \
  --artifacts models/llama31_tt/chat_artifacts \
  --runner models/llama31_tt/llama31_chat_run.py \
  --max-cache-len 1024 \
  -o build/models/llama31_tt/llama31_tt.rhal.mlir

build/bin/rax-pack build/models/llama31_tt/llama31_tt.rhal.mlir \
  -o build/models/llama31_tt/llama31_tt.rax
```

For perf-only decode runs whose TTNN decode flatbuffer already returns an
integer token id, add the manifest flags consumed by `buddy-cli`:

```bash
python3 tools/buddy-codegen/gen_tenstorrent_manifest.py \
  --prefill-ttnn models/llama31_tt/ttir_out_static/llama31_prefill_static_argattrs_argmax.ttnn \
  --decode-ttnn models/llama31_tt/ttir_out_static/llama31_decode_static_argattrs_argmax.ttnn \
  --artifacts models/llama31_tt/chat_artifacts_argmax \
  --runner models/llama31_tt/llama31_chat_run.py \
  --max-cache-len 1024 \
  --device-token-loop \
  --ignore-eos \
  -o build/models/llama31_tt/llama31_tt_argmax.rhal.mlir
```

Then:

```bash
BUDDY_TT_PYTHON=python3 \
build/bin/buddy-cli \
  --model build/models/llama31_tt/llama31_tt.rax \
  --prompt "Hello" \
  --max-tokens 32
```

To run the full Llama 3.1 8B path from capture through `buddy-cli`:

```bash
printf 'Hello, who are you?\n' | \
BUDDY_TT_CONDA_ENV=buddy-ttmlir-p150a \
BUDDY_BUILD=/tmp/buddy-build-tt-p150a \
TTMLIR_SOURCE=$PWD/thirdparty/tt-mlir \
TTMLIR_BUILD=/tmp/buddy-ttmlir-build \
HF_HOME=$HOME/.cache/huggingface \
HUGGINGFACE_HUB_CACHE=$HOME/.cache/huggingface/hub \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
MAX_CACHE_LEN=1024 \
MAX_NEW_TOKENS=4 \
RUN_WITH_BUDDY_CLI=1 \
models/llama31_tt/run_llama31_p150_chat.sh
```

## Notes

- `.ttnn` flatbuffers should be generated and run with the same `tt-mlir/ttrt`
  build. Version skew can cause runtime failures.
- System descriptors are hardware-specific. Regenerate flatbuffers when moving
  between machines or different Tenstorrent configurations.
- The current `llama31_tt` runner bridges to the Python `ttrt.runtime` runner.
  A later native C++ runtime runner can reuse the same `.rax` artifact contract.
