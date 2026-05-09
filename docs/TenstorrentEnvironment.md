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
cd "$BUDDY_REPO_ROOT"

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
cd "$BUDDY_REPO_ROOT"

conda create -n buddy-ttmlir-p150a python=3.12 -y
conda activate buddy-ttmlir-p150a
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
`buddy-cli`. Generated model artifacts are written under
`$BUDDY_BUILD/models/llama31_tt`. The generated `.rax` is self-contained by
default: it embeds the TTNN flatbuffers, Python runner, and Llama chat artifact
files including the weight archives.
Keep this path under `$BUDDY_BUILD/models/llama31_tt`; the self-contained
package needs enough free space for another copy of the Llama artifact payload.
The prefill and decode phases share one embedded weight archive.

By default the scripts use the Hugging Face model id
`meta-llama/Llama-3.1-8B-Instruct`. With `HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1`, the model and tokenizer must already exist in the
local Hugging Face cache. To use an explicit local checkout, set
`LLAMA31_MODEL_PATH` to the model directory. The same variable is also needed
when running a generated `.rax` directly with `buddy-cli`, because the package
embeds the model weights but still uses the Hugging Face tokenizer at runtime.

```bash
cd "$BUDDY_REPO_ROOT"

printf 'Hello, who are you?\n' | \
BUDDY_TT_CONDA_ENV=buddy-ttmlir-p150a \
BUDDY_BUILD=$BUDDY_REPO_ROOT/build-tt-p150a \
TTMLIR_TOOLCHAIN_DIR=$BUDDY_REPO_ROOT/build-ttmlir-toolchain \
TTMLIR_SOURCE=$BUDDY_REPO_ROOT/thirdparty/tt-mlir \
TTMLIR_BUILD=$BUDDY_REPO_ROOT/build-ttmlir \
BUDDY_LLAMA31_ARTIFACT_ROOT=$BUDDY_BUILD/models/llama31_tt \
LLAMA31_MODEL_PATH=/path/to/Llama-3.1-8B-Instruct \
HF_HOME=$HOME/.cache/huggingface \
HUGGINGFACE_HUB_CACHE=$HOME/.cache/huggingface/hub \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
MAX_CACHE_LEN=1024 \
MAX_NEW_TOKENS=4 \
RUN_WITH_BUDDY_CLI=1 \
models/llama31_tt/run_llama31_p150_chat.sh
```

If `LLAMA31_MODEL_PATH` is omitted, the default Hugging Face id is used. In
offline mode that succeeds only when the model has already been cached locally;
without offline mode, Transformers may try to download it from Hugging Face and
will require access to the gated Llama 3.1 repository.
