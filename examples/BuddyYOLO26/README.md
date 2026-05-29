# Buddy Compiler YOLO26n Example (Verified Setup Guide)

This document reflects a fully tested setup process on Ubuntu 20.04 with recent LLVM/MLIR and current buddy-mlir sources.

---

# 1. Create and activate Python environment

We recommend using conda.

```bash
conda create -n buddy python=3.11 -y
conda activate buddy
```

Install Python dependencies:

```bash
cd buddy-mlir
pip install -r requirements.txt
```

Install additional required packages:

```bash
pip install pybind11 nanobind ultralytics
```

---

# 2. Install system dependencies

```bash
sudo apt update
sudo apt install -y \
    build-essential \
    ninja-build \
    clang \
    lld \
    git \
    wget \
    curl \
    libpng-dev \
    libjpeg-dev \
    zlib1g-dev \
    pkg-config \
    python3-dev
```

If Ubuntu ships an old CMake (< 3.20), install newer CMake:

```bash
conda install -c conda-forge cmake ninja
```

Verify:

```bash
cmake --version
```

CMake >= 3.20 is required.

---

# 3. Clone buddy-mlir correctly

Do NOT download ZIP archives.

The repository depends on git submodules.

Correct method:

```bash
git clone --recursive https://github.com/buddy-compiler/buddy-mlir.git
```

If already cloned without submodules:

```bash
cd buddy-mlir
git submodule update --init --recursive
```

Verify LLVM source exists:

```bash
ls llvm/llvm
```

---

# 4. Build LLVM/MLIR

```bash
cd buddy-mlir

mkdir -p llvm/build
cd llvm/build
```

Configure:

```bash
cmake -G Ninja ../llvm \
    -DLLVM_ENABLE_PROJECTS="mlir;clang" \
    -DLLVM_ENABLE_RUNTIMES="openmp" \
    -DLLVM_TARGETS_TO_BUILD="host;RISCV" \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DOPENMP_ENABLE_LIBOMPTARGET=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
    -DPython3_EXECUTABLE=$(which python3)
```

Build:

```bash
ninja check-clang check-mlir
```

Notes:

* `LLVM_ENABLE_PROJECTS="mlir;clang;openmp"` is obsolete on recent LLVM.
* `omp` ninja target no longer exists.

---

# 5. Build buddy-mlir

```bash
cd ~/buddy-mlir

mkdir -p build
cd build
```

Configure:

```bash
cmake -G Ninja .. \
    -DMLIR_DIR=$HOME/buddy-mlir/llvm/build/lib/cmake/mlir \
    -DLLVM_DIR=$HOME/buddy-mlir/llvm/build/lib/cmake/llvm \
    -Dpybind11_DIR=$(python3 -m pybind11 --cmakedir) \
    -Dnanobind_DIR=$(python3 -c "import nanobind; print(nanobind.cmake_dir())") \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUDDY_MLIR_ENABLE_PYTHON_PACKAGES=ON \
    -DBUDDY_MLIR_ENABLE_DIP_LIB=ON \
    -DBUDDY_ENABLE_PNG=ON \
    -DBUDDY_YOLO26_EXAMPLES=ON \
    -DPython3_EXECUTABLE=$(which python3)
```

Build:

```bash
ninja
```

---

# 6. Configure PYTHONPATH

```bash
cd ~/buddy-mlir/build

export BUDDY_MLIR_BUILD_DIR=$PWD
export LLVM_MLIR_BUILD_DIR=$PWD/../llvm/build

export PYTHONPATH=\
${LLVM_MLIR_BUILD_DIR}/tools/mlir/python_packages/mlir_core:\
${BUDDY_MLIR_BUILD_DIR}/python_packages
```

Verify:

```bash
python3 -c "from buddy.compiler.frontend import DynamoCompiler"
```

---

# 7. Build YOLO26 example

```bash
cd ~/buddy-mlir/build

cmake ..
ninja buddy-yolo26n-run
```

The importer automatically downloads `yolo26n.pt` if not found.

Optional:

```bash
export YOLO26N_MODEL_PATH=/path/to/yolo26n.pt
```

---

# 8. Run inference

Example:

```bash
./bin/buddy-yolo26n-run \
~/buddy-mlir/examples/BuddyYOLO26/images/bus_16bit.bmp
```

Expected output:

```text
YOLO26n Inference Powered by Buddy Compiler

[Log] Inference time: ...
[0] class_id=0 label=person ...
[3] class_id=5 label=bus ...
Detections: 5
```

---

# Common Issues

## 1. `MLIRConfig.cmake` not found

LLVM/MLIR was not built correctly.

Verify:

```bash
find ~/buddy-mlir/llvm/build -name MLIRConfig.cmake
```

---

## 2. `No module named ultralytics`

Install:

```bash
pip install ultralytics
```

---

## 3. `No module named buddy.compiler.frontend`

Usually caused by incorrect `PYTHONPATH`.

Verify:

```bash
export PYTHONPATH=$HOME/buddy-mlir/build/python_packages:$HOME/buddy-mlir/llvm/build/tools/mlir/python_packages/mlir_core
```

Test:

```bash
python3 -c "from buddy.compiler.frontend import DynamoCompiler"
```

---

## 4. `Could NOT find PNG`

Install:

```bash
sudo apt install libpng-dev
```

---

## 5. `Could NOT find JPEG`

Install:

```bash
sudo apt install libjpeg-dev
```

---

# Verified Environment

* Ubuntu 20.04
* Python 3.11
* CMake 4.x
* Ninja 1.13
* LLVM/MLIR main branch
* buddy-mlir current main branch

---

# Validation Status

Successfully validated:

* LLVM build
* MLIR build
* buddy-mlir build
* Python bindings
* YOLO26 importer
* MLIR lowering
* object generation
* executable linking
* runtime inference
* detection output correctness

Architecture tested:

* x86_64 Linux
