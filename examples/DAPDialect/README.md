# Buddy Compiler RFFT Example

## Introduction

This example demonstrates using the RFFT operator in Buddy Compiler to compute the Fourier transform of real numbers, with the algorithm derived from the [PocketFFT library](https://gitlab.mpcdf.mpg.de/mtr/pocketfft/-/tree/master?ref_type=heads).

## How to run


1. Build and check LLVM/MLIR

```
$ cd buddy-mlir
$ mkdir llvm/build
$ cd llvm/build
$ cmake -G Ninja ../llvm \
    -DLLVM_ENABLE_PROJECTS="mlir;clang;openmp" \
    -DLLVM_TARGETS_TO_BUILD="host;RISCV" \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DOPENMP_ENABLE_LIBOMPTARGET=OFF \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
    -DPython3_EXECUTABLE=$(which python3)
$ ninja check-clang check-mlir omp
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
    -DBUDDY_EXAMPLES=ON \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DBUDDY_MLIR_ENABLE_PYTHON_PACKAGES=ON \
    -DPython3_EXECUTABLE=$(which python3)
$ ninja
$ ninja check-buddy
```


3. Run the RFFT example
```bash
$ cd bin
$ ./buddy-rfft
```
The result will be saved in the ``whisperPreprocessResultRFFT.txt`` and you can edit the input data in ``buddy-mlir/examples/DAPDialect/RFFT.cpp``

4. Enjoy it!
