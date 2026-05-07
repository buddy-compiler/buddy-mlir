# Buddy Llama 3.1 8B on Tenstorrent P150A

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
export BUDDY_BUILD="$BUDDY_REPO_ROOT/build-tt-p150a"
```

## Initialize tt-mlir

The `git submodule update --init` command is the step that downloads
`thirdparty/tt-mlir` when it is not already present. The extra config and tag
fetch keep the checkout non-shallow so tt-mlir's CMake version detection works.

```bash
git submodule sync thirdparty/tt-mlir
git config submodule.thirdparty/tt-mlir.shallow false
git submodule update --init thirdparty/tt-mlir
git -C thirdparty/tt-mlir fetch --tags --force
```

## Create Python Environment

Use one conda environment to bootstrap the tt-mlir toolchain and Python
bindings. After the toolchain venv is created, install the Llama frontend
packages there as well.

```bash
conda create -n buddy-ttmlir-p150a python=3.12 -y
conda activate buddy-ttmlir-p150a
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

python -m pip install --upgrade pip
python -m pip install cmake ninja
python -m pip install \
  -r thirdparty/tt-mlir/env/build-requirements.txt \
  -r thirdparty/tt-mlir/env/ttnn-requirements.txt \
  -r thirdparty/tt-mlir/test/python/requirements.txt \
  -r thirdparty/tt-mlir/tools/ttrt/requirements.txt
conda install -c conda-forge doxygen graphviz -y
```

## Build Buddy LLVM/MLIR

Buddy is built against the LLVM/MLIR submodule in this repository. The
Tenstorrent toolchain is used later for `ttmlir-opt`, `ttmlir-translate`, and
`ttrt`, but not as Buddy's `MLIR_DIR` / `LLVM_DIR`.

```bash
git submodule update --init llvm

mkdir -p "$BUDDY_LLVM_BUILD"
cd "$BUDDY_LLVM_BUILD"

cmake -G Ninja "$BUDDY_REPO_ROOT/llvm/llvm" \
  -DLLVM_ENABLE_PROJECTS="mlir;clang;openmp" \
  -DLLVM_TARGETS_TO_BUILD="host;RISCV" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DOPENMP_ENABLE_LIBOMPTARGET=OFF \
  -DCMAKE_BUILD_TYPE=RELEASE \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DPython3_EXECUTABLE=$(which python3)

ninja check-clang check-mlir omp
```

## Build tt-mlir Toolchain

This builds the upstream LLVM/MLIR-based toolchain used by tt-mlir.

```bash
mkdir -p "$TTMLIR_TOOLCHAIN_DIR"

cmake -G Ninja -S thirdparty/tt-mlir/env -B "$TTMLIR_ENV_BUILD" \
  -DCMAKE_C_COMPILER=/usr/bin/clang-20 \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++-20 \
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

cmake -G Ninja -S . -B "$TTMLIR_BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/usr/bin/clang-20 \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++-20 \
  -DTTMLIR_ENABLE_RUNTIME=ON \
  -DTT_RUNTIME_ENABLE_PERF_TRACE=ON \
  -DTTMLIR_ENABLE_TESTS=OFF \
  -DTTMLIR_ENABLE_EXPLORER=OFF \
  -DTTMLIR_ENABLE_ALCHEMIST=OFF \
  -DCMAKE_CXX_FLAGS=-Wno-error=deprecated-declarations

cmake --build "$TTMLIR_BUILD" \
  --target ttmlir-opt ttmlir-translate TTMLIRPythonModules TTRTPythonModules ttrt

"$TTMLIR_TOOLCHAIN_DIR/venv/bin/python" -m pip install --force-reinstall --no-deps \
  "$TTMLIR_BUILD"/tools/ttrt/build/ttrt-*.whl

"$TTMLIR_TOOLCHAIN_DIR/venv/bin/python" -c "import ttrt, ttrt.runtime; print('ttrt ok')"
```

## Build Buddy

This configures Buddy with Tenstorrent checks enabled and builds `buddy-cli`
plus `rax-pack`.

```bash
cd "$BUDDY_REPO_ROOT"
source thirdparty/tt-mlir/env/activate
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

## Smoke Test

Run a small import/query check before launching the full model.

```bash
cd "$BUDDY_REPO_ROOT"
source thirdparty/tt-mlir/env/activate
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$TTMLIR_BUILD/python_packages:${PYTHONPATH:-}"

"$TTMLIR_TOOLCHAIN_DIR/venv/bin/python" -c "import ttrt, ttrt.runtime, ttmlir.ir; print('tt runtime ok')"
"$TTMLIR_TOOLCHAIN_DIR/venv/bin/python" -m ttrt query
```

## Run Llama 3.1 8B

This command captures, lowers, packages, and runs the Llama 3.1 8B flow through
`buddy-cli`.

```bash
cd "$BUDDY_REPO_ROOT"

printf 'Hello, who are you?\n' | \
BUDDY_TT_CONDA_ENV=buddy-ttmlir-p150a \
BUDDY_BUILD=$BUDDY_REPO_ROOT/build-tt-p150a \
TTMLIR_SOURCE=$BUDDY_REPO_ROOT/thirdparty/tt-mlir \
TTMLIR_BUILD=$BUDDY_REPO_ROOT/build-ttmlir \
HF_HOME=$HOME/.cache/huggingface \
HUGGINGFACE_HUB_CACHE=$HOME/.cache/huggingface/hub \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
MAX_CACHE_LEN=1024 \
MAX_NEW_TOKENS=4 \
RUN_WITH_BUDDY_CLI=1 \
models/llama31_tt/run_llama31_p150_chat.sh
```
