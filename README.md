# BUDDY MLIR

An MLIR-Based Ideas Landing Project.

## Getting Started

### LLVM/MLIR Dependencies

This project uses LLVM/MLIR as an external library. Please make sure [the dependencies](https://mlir.llvm.org/getting_started/) are available
on your machine.

### Clone and Initialize


```
$ git clone git@github.com:buddy-compiler/buddy-mlir.git
$ cd buddy-mlir
$ git submodule update --init
```

### Build and Test LLVM/MLIR

```
$ cd buddy-mlir
$ mkdir llvm/build
$ cd llvm/build
$ cmake -G Ninja ../llvm -DLLVM_ENABLE_PROJECTS="mlir" \
-DLLVM_TARGETS_TO_BUILD="host" \
-DLLVM_ENABLE_ASSERTIONS=ON \
-DCMAKE_BUILD_TYPE=RELEASE
$ ninja
$ ninja check-mlir
```

### Build buddy-mlir

```
$ cd buddy-mlir
$ mkdir build
$ cd build
$ cmake -G Ninja .. \
    -DMLIR_DIR=$PWD/../llvm/build/lib/cmake/mlir \
    -DLLVM_DIR=$PWD/../llvm/build/lib/cmake/llvm \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=RELEASE
$ ninja
```
