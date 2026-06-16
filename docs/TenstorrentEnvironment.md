# Buddy Tenstorrent Environment

This guide sets up the optional Tenstorrent toolchain used by Buddy:
Buddy LLVM/MLIR, upstream `tt-mlir`, TTNN flatbuffer tooling, and the `ttrt`
runtime package. Model-specific package generation and `buddy-cli` commands
live in the corresponding model README files.

## Setup Paths

Run all commands from the Buddy repository root. Build directories stay inside
the checkout and are ignored by Git.

```bash
cd /path/to/buddy-mlir

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
bindings.

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

This configures and builds Buddy with Tenstorrent support enabled. Model-specific
CMake flags and package targets are documented in each model README.

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
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_BUILD_TYPE=RELEASE \
  -DBUDDY_MLIR_ENABLE_PYTHON_PACKAGES=ON \
  -DPython3_EXECUTABLE=$TTMLIR_TOOLCHAIN_DIR/venv/bin/python \
  -DBUDDY_ENABLE_TENSTORRENT=ON \
  -DBUDDY_TT_MLIR_SOURCE_DIR=$BUDDY_REPO_ROOT/thirdparty/tt-mlir \
  -DBUDDY_TT_MLIR_BUILD_DIR=$TTMLIR_BUILD

cmake --build "$BUDDY_BUILD"
cmake --build "$BUDDY_BUILD" --target check-buddy
```

## Smoke Test

Run a small import/query check before launching model-specific package targets
or `buddy-cli`.

```bash
cd "$BUDDY_REPO_ROOT/thirdparty/tt-mlir"
source env/activate
cd "$BUDDY_REPO_ROOT"

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$TTMLIR_BUILD/python_packages:${PYTHONPATH:-}"

"$TTMLIR_TOOLCHAIN_DIR/venv/bin/python" -c "import ttrt, ttrt.runtime, ttmlir.ir; print('tt runtime ok')"
"$TTMLIR_TOOLCHAIN_DIR/venv/bin/python" -m ttrt query
```

To suppress TT-Metal, UMD, and tt-mlir runtime info/debug logs, add these
before running package-generation or runtime commands.

```bash
export TT_LOGGER_LEVEL=FATAL
export TT_METAL_LOGGER_LEVEL=FATAL
export TTMLIR_RUNTIME_LOGGER_LEVEL=FATAL
```

For model builds, continue with the README under `models/<model_name>/`.
