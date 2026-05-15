# Buddy Llama 3.1 8B on Tenstorrent

This guide sets up the optional Tenstorrent path for running Llama 3.1 8B
through `buddy-cli`, `tt-mlir`, TTNN flatbuffers, and `ttrt`.

## Setup Paths

Run all commands from the Buddy repository root. Build directories stay inside
the checkout and are ignored by Git.

```bash
cd ~/buddy-mlir

export BUDDY_REPO_ROOT=$(pwd)
export BUDDY_LLVM_BUILD="$BUDDY_REPO_ROOT/llvm/build"
export TTMLIR_TOOLCHAIN_DIR="$BUDDY_REPO_ROOT/build-ttmlir-toolchain"
export TTMLIR_ENV_BUILD="$BUDDY_REPO_ROOT/build-ttmlir-env"
export TTMLIR_BUILD="$BUDDY_REPO_ROOT/build-ttmlir"
export BUDDY_BUILD="$BUDDY_REPO_ROOT/build-tenstorrent"
```

## Initialize tt-mlir

The `git submodule update --init` command is the step that downloads
`thirdparty/tt-mlir` when it is not already present. The runtime build below
also passes an explicit package version so the build does not depend on
`git describe`.

```bash
cd "$BUDDY_REPO_ROOT"

git submodule sync thirdparty/tt-mlir
git submodule update --init thirdparty/tt-mlir
```

## Create Python Environment

Use one conda environment to bootstrap the tt-mlir toolchain and Python
bindings. After the toolchain venv is created, install the Llama frontend
packages there as well.

```bash
cd "$BUDDY_REPO_ROOT"

conda create -n buddy-ttmlir python=3.12 -y
conda activate buddy-ttmlir
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

python -m pip install --upgrade pip
python -m pip install cmake ninja
python -m pip install "pybind11>=2.10" nanobind numpy pyyaml pygments
python -m pip install \
  -r thirdparty/tt-mlir/env/build-requirements.txt \
  -r thirdparty/tt-mlir/env/ttnn-requirements.txt \
  -r thirdparty/tt-mlir/test/python/requirements.txt \
  -r thirdparty/tt-mlir/tools/ttrt/requirements.txt
python -c "import pybind11; print(pybind11.get_cmake_dir())"
conda install -c conda-forge doxygen graphviz -y
```

## Build Buddy LLVM/MLIR

Buddy is built against the LLVM/MLIR submodule in this repository. The
Tenstorrent toolchain is used later for `ttmlir-opt`, `ttmlir-translate`, and
`ttrt`, but not as Buddy's `MLIR_DIR` / `LLVM_DIR`. This follows the standard
Buddy build style and lets CMake use the normal host compiler for the first
LLVM build.

```bash
cd "$BUDDY_REPO_ROOT"

git submodule update --init llvm

mkdir -p "$BUDDY_LLVM_BUILD"

cmake -G Ninja -S "$BUDDY_REPO_ROOT/llvm/llvm" -B "$BUDDY_LLVM_BUILD" \
  -DLLVM_ENABLE_PROJECTS="mlir;clang;openmp" \
  -DLLVM_TARGETS_TO_BUILD="host;RISCV" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DOPENMP_ENABLE_LIBOMPTARGET=OFF \
  -DCMAKE_BUILD_TYPE=RELEASE \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DPython3_EXECUTABLE=$(which python3) \
  -Dpybind11_DIR="$(python -m pybind11 --cmakedir)"

cmake --build "$BUDDY_LLVM_BUILD" --target check-clang check-mlir omp

export PATH="$BUDDY_LLVM_BUILD/bin:$PATH"
```

## Build tt-mlir Toolchain

This builds the upstream LLVM/MLIR-based toolchain used by tt-mlir. The
`CC=clang CXX=clang++` prefix uses the Buddy-built clang from
`$BUDDY_LLVM_BUILD/bin`, because that directory was added to `PATH` above. No
system `/usr/bin/clang-20` path is required.

```bash
cd "$BUDDY_REPO_ROOT"

mkdir -p "$TTMLIR_TOOLCHAIN_DIR"

CC=clang CXX=clang++ \
cmake -G Ninja -S "$BUDDY_REPO_ROOT/thirdparty/tt-mlir/env" -B "$TTMLIR_ENV_BUILD" \
  -DLLVM_BUILD_TYPE=MinSizeRel

cmake --build "$TTMLIR_ENV_BUILD"
```

The tt-mlir activation script puts `$TTMLIR_TOOLCHAIN_DIR/venv/bin` first in
`PATH`. Install the Llama frontend packages into that Python environment too.

```bash
"$TTMLIR_TOOLCHAIN_DIR/venv/bin/python" -m pip install --index-url https://download.pytorch.org/whl/cpu torch
"$TTMLIR_TOOLCHAIN_DIR/venv/bin/python" -m pip install \
  transformers accelerate safetensors sentencepiece
"$TTMLIR_TOOLCHAIN_DIR/venv/bin/python" -c "import torch, transformers; print('llama frontend deps ok')"
```

## Build tt-mlir Runtime

This builds `ttmlir-opt`, `ttmlir-translate`, Python modules, and the `ttrt`
runtime package used to execute TTNN flatbuffers.

```bash
cd "$BUDDY_REPO_ROOT/thirdparty/tt-mlir"
source env/activate
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$TTMLIR_BUILD/python_packages:${PYTHONPATH:-}"

CC=clang CXX=clang++ \
cmake -G Ninja -S . -B "$TTMLIR_BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DTTMLIR_ENABLE_RUNTIME=ON \
  -DTT_RUNTIME_ENABLE_PERF_TRACE=ON \
  -DTTMLIR_ENABLE_TESTS=OFF \
  -DTTMLIR_ENABLE_EXPLORER=OFF \
  -DTTMLIR_ENABLE_ALCHEMIST=OFF \
  -DCMAKE_CXX_FLAGS=-Wno-error=deprecated-declarations \
  -DTTMLIR_VERSION_MAJOR=0 \
  -DTTMLIR_VERSION_MINOR=1 \
  -DTTMLIR_VERSION_PATCH=0

cmake --build "$TTMLIR_BUILD" \
  --target ttmlir-opt ttmlir-translate TTMLIRPythonModules TTRTPythonModules ttrt

"$TTMLIR_TOOLCHAIN_DIR/venv/bin/python" -m pip install --force-reinstall --no-deps \
  "$TTMLIR_BUILD"/tools/ttrt/build/ttrt-*.whl

"$TTMLIR_TOOLCHAIN_DIR/venv/bin/python" -c "import ttrt, ttrt.runtime; print('ttrt ok')"
```

## Build Buddy

This configures Buddy with Tenstorrent checks enabled and builds the host
tools. The Llama package itself is generated by the `llama31_tt_rax` target in
the next section.

```bash
cd "$BUDDY_REPO_ROOT/thirdparty/tt-mlir"
source env/activate
cd "$BUDDY_REPO_ROOT"

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$TTMLIR_BUILD/python_packages:${PYTHONPATH:-}"

mkdir -p "$BUDDY_BUILD"

cmake -G Ninja -S "$BUDDY_REPO_ROOT" -B "$BUDDY_BUILD" \
  -DMLIR_DIR=$BUDDY_LLVM_BUILD/lib/cmake/mlir \
  -DLLVM_DIR=$BUDDY_LLVM_BUILD/lib/cmake/llvm \
  -DPython3_EXECUTABLE=$TTMLIR_TOOLCHAIN_DIR/venv/bin/python \
  -DBUDDY_MLIR_ENABLE_PYTHON_PACKAGES=ON \
  -DBUDDY_BUILD_LLAMA31_TT_MODEL=ON \
  -DBUDDY_ENABLE_TENSTORRENT=ON \
  -DBUDDY_TT_MLIR_SOURCE_DIR=$BUDDY_REPO_ROOT/thirdparty/tt-mlir \
  -DBUDDY_TT_MLIR_BUILD_DIR=$TTMLIR_BUILD

cmake --build "$BUDDY_BUILD"
```

## Build Llama 3.1 Package

This target captures the PyTorch graph, lowers it to TTNN flatbuffers, prepares
`chat_artifacts`, and writes a self-contained package to
`$BUDDY_BUILD/models/llama31_tt/llama31_tt.rax`.

```bash
cd "$BUDDY_REPO_ROOT/thirdparty/tt-mlir"
source env/activate
cd "$BUDDY_REPO_ROOT"

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$TTMLIR_BUILD/python_packages:${PYTHONPATH:-}"

LLAMA31_MODEL_PATH=/path/to/Llama-3.1-8B-Instruct \
HF_HOME=$HOME/.cache/huggingface \
HUGGINGFACE_HUB_CACHE=$HOME/.cache/huggingface/hub \
cmake --build "$BUDDY_BUILD" --target llama31_tt_rax
```

## Smoke Test

Run a small import/query check before launching the full model.

```bash
cd "$BUDDY_REPO_ROOT/thirdparty/tt-mlir"
source env/activate
cd "$BUDDY_REPO_ROOT"

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$TTMLIR_BUILD/python_packages:${PYTHONPATH:-}"

"$TTMLIR_TOOLCHAIN_DIR/venv/bin/python" -c "import ttrt, ttrt.runtime, ttmlir.ir; print('tt runtime ok')"
"$TTMLIR_TOOLCHAIN_DIR/venv/bin/python" -m ttrt query
```

## Run Llama 3.1 8B

After `llama31_tt_rax` has completed, serving can call `buddy-cli` directly.
The package embeds the TTNN flatbuffers and Llama chat artifact files,
including the shared weight archive. Set `LLAMA31_MODEL_PATH` to a local model
directory when running `buddy-cli`; the package embeds model weights but the
native C++ runtime still reads tokenizer files from that directory.

```bash
source "$BUDDY_REPO_ROOT/thirdparty/tt-mlir/env/activate"
cd "$BUDDY_REPO_ROOT"

ulimit -v 95000000
ulimit -m 95000000
export TT_METAL_RUNTIME_ROOT="$BUDDY_REPO_ROOT/thirdparty/tt-mlir/third_party/tt-metal/src/tt-metal"
export TT_METAL_HOME="$TT_METAL_RUNTIME_ROOT"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$TTMLIR_BUILD/python_packages:${PYTHONPATH:-}"
export BUDDY_RAX_PAYLOAD_DIR=/tmp/$USER/buddy_rax_payload
mkdir -p "$BUDDY_RAX_PAYLOAD_DIR"

LLAMA31_MODEL_PATH=/path/to/Llama-3.1-8B-Instruct \
"$BUDDY_BUILD/bin/buddy-cli" \
  --model "$BUDDY_BUILD/models/llama31_tt/llama31_tt.rax" \
  --prompt "Hello, who are you?" \
  --max-tokens 32 \
  --repeat-penalty 1.12 \
  --repeat-last-n 128
```

To suppress TT-Metal, UMD, and tt-mlir runtime info/debug logs, add these
before running `buddy-cli`.

```bash
export TT_LOGGER_LEVEL=FATAL
export TT_METAL_LOGGER_LEVEL=FATAL
export TTMLIR_RUNTIME_LOGGER_LEVEL=FATAL
```

`LLAMA31_MODEL_PATH` must point at a local Llama-3.1-8B-Instruct checkout that
contains `original/tokenizer.model` or `tokenizer.model`. The generated `.rax`
embeds the TTNN flatbuffers and weights, but the native C++ runtime still needs
the tokenizer files and does not download from Hugging Face.

For end-to-end development, `models/llama31_tt/run_llama31_tt_chat.sh` still
supports the full capture, lower, package, and interactive run flow. Set
`PACKAGE_ONLY=1` when using that script only to generate the `.rax` package.
