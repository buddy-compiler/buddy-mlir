# Buddy Compiler Qwen3 Example

## Introduction

This example shows how to use Buddy Compiler to compile a Qwen3 model to MLIR code then run it.

## How to run on non-RISC-V device

0. Enter Python virtual environment.

We recommend you to use anaconda3 to create python virtual environment. You should install python packages as buddy-mlir/requirements.

```
$ conda activate <your virtual environment name>
$ cd buddy-mlir
$ pip install -r requirements.txt
```

1. Build and check LLVM/MLIR

```
$ cd buddy-mlir
$ mkdir llvm/build
$ cd llvm/build
$ cmake -G Ninja ../llvm \
    -DLLVM_ENABLE_PROJECTS="mlir;clang" \
    -DLLVM_TARGETS_TO_BUILD="host;RISCV" \
    -DLLVM_ENABLE_RUNTIMES="openmp" \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DOPENMP_ENABLE_LIBOMPTARGET=OFF \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
    -DPython3_EXECUTABLE=$(which python3)
$ ninja check-clang check-mlir check-omp
```

2. Build and check buddy-mlir

```
$ cd buddy-mlir
$ mkdir build
$ cd build
$ cmake -G Ninja .. \
    -DMLIR_DIR=$PWD/../llvm/build/lib/cmake/mlir \
    -DLLVM_DIR=$PWD/../llvm/build/lib/cmake/llvm \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DBUDDY_MLIR_ENABLE_PYTHON_PACKAGES=ON \
    -DPython3_EXECUTABLE=$(which python3)
$ ninja
$ ninja check-buddy
```

Set the `PYTHONPATH` environment variable. Make sure that the `PYTHONPATH` variable includes the directory of LLVM/MLIR python bindings and the directory of Buddy MLIR python packages.

```bash
$ export PYTHONPATH=/path-to-buddy-mlir/llvm/build/tools/mlir/python_packages/mlir_core:/path-to-buddy-mlir/build/python_packages:${PYTHONPATH}

// For example:
// Navigate to your buddy-mlir/build directory
$ cd buddy-mlir/build
$ export BUDDY_MLIR_BUILD_DIR=$PWD
$ export LLVM_MLIR_BUILD_DIR=$PWD/../llvm/build
$ export PYTHONPATH=${LLVM_MLIR_BUILD_DIR}/tools/mlir/python_packages/mlir_core:${BUDDY_MLIR_BUILD_DIR}/python_packages:${PYTHONPATH}
```

3. Set model environment variable.

```bash
$ export Qwen3_0_6B_MODEL_PATH=/path-to-qwen3-model/

// For example:
$ export QWEN3_0_6B_MODEL_PATH=/home/xxx/QWEN3_0_6B_MODEL
```
Alternatively, you can leave the path blank, and import-qwen3.py will automatically download the model for you.

4. Build and run the QWEN3 example

```bash
$ cmake -G Ninja .. -DBUDDY_QWEN3_EXAMPLES=ON

//f32
$ ninja buddy-qwen3-0.6b-run
$ ./bin/buddy-qwen3-0.6b-run

// NUMA node binding
numactl --cpunodebind=0,1,2,3 --interleave=0,1,2,3 taskset -c 0-47 ./bin/buddy-qwen3-0.6b-run
```

5. Enjoy it!
