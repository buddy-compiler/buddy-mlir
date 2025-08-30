# Buddy Compiler LeNet Example

## Train the LeNet Model

Activate your python environment.

```bash
$ cd buddy-mlir
$ cd examples/BuddyLeNet
$ python pytorch-lenet-train.py
```

## LeNet Model Inference

### Activate your python environment.

```bash
$ conda activate <your env>
```

### Build LLVM

```bash
$ cd buddy-mlir
$ mkdir llvm/build
$ cd llvm/build

// CPU
$ cmake -G Ninja ../llvm \
    -DLLVM_ENABLE_PROJECTS="mlir;clang;openmp" \
    -DLLVM_TARGETS_TO_BUILD="host;RISCV" \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DOPENMP_ENABLE_LIBOMPTARGET=OFF \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
    -DPython3_EXECUTABLE=$(which python3)

// GPU
$ cmake -G Ninja ../llvm \
    -DLLVM_ENABLE_PROJECTS="mlir;clang;openmp" \
    -DLLVM_TARGETS_TO_BUILD="host;RISCV;NVPTX" \
    -DMLIR_ENABLE_CUDA_RUNNER=ON \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DOPENMP_ENABLE_LIBOMPTARGET=OFF \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
    -DPython3_EXECUTABLE=$(which python3)

$ ninja check-clang check-mlir omp
```

### Build buddy-mlir

```bash
$ cd buddy-mlir
$ mkdir build && cd build
$ cmake -G Ninja .. \
    -DMLIR_DIR=$PWD/../llvm/build/lib/cmake/mlir \
    -DLLVM_DIR=$PWD/../llvm/build/lib/cmake/llvm \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DBUDDY_MLIR_ENABLE_PYTHON_PACKAGES=ON \
    -DPython3_EXECUTABLE=$(which python3) \
    -DBUDDY_MLIR_ENABLE_DIP_LIB=ON \
    -DBUDDY_ENABLE_PNG=ON
$ ninja
$ ninja check-buddy
```

### Set the `PYTHONPATH` environment variable.

Make sure you are in the build directory.

```bash
$ export BUDDY_MLIR_BUILD_DIR=$PWD
$ export LLVM_MLIR_BUILD_DIR=$PWD/../llvm/build
$ export PYTHONPATH=${LLVM_MLIR_BUILD_DIR}/tools/mlir/python_packages/mlir_core:${BUDDY_MLIR_BUILD_DIR}/python_packages:${PYTHONPATH}
```

### Build and run the LeNet example

```bash
$ cmake -G Ninja .. -DBUDDY_LENET_EXAMPLES=ON

// CPU
$ ninja buddy-lenet-run
$ cd bin
$ ./buddy-lenet-run

// GPU
$ ninja buddy-lenet-run-gpu
$ cd bin
$ ./buddy-lenet-run-gpu
```

## Debug the Lowering Pass Pipeline with Fake Parameters.

```bash
$ cd buddy-mlir
$ cd examples/BuddyLeNet
$ make buddy-lenet-lower
$ make buddy-lenet-translate
$ make buddy-lenet-run
```
