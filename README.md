# BUDDY MLIR

MLIR-Based Ideas Landing Project ([Project page](https://buddy-compiler.github.io/)).

## Getting Started

Two building strategies are provided: one-step building strategy and two-step building strategy. The one-step building strategy uses buddy-mlir as an external library, while the two-step building strategy uses LLVM/MLIR as an external library. 

### LLVM/MLIR Dependencies

Before building, please make sure [the dependencies](https://llvm.org/docs/GettingStarted.html#requirements) are available
on your machine.

### Clone and Initialize

```
$ git clone git@github.com:buddy-compiler/buddy-mlir.git
$ cd buddy-mlir
$ git submodule update --init
```

### One-step building strategy

If you have not previously built llvm-project and you want to use buddy-mlir as an external library, you can follow these commands to build llvm-project as well as buddy-mlir.

```
$ cmake -G Ninja -Bbuild \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_PROJECTS="mlir;clang" \
    -DLLVM_TARGETS_TO_BUILD="host;RISCV" \
    -DLLVM_EXTERNAL_PROJECTS="buddy-mlir" \
    -DLLVM_EXTERNAL_BUDDY_MLIR_SOURCE_DIR="$PWD" \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=RELEASE \
    llvm/llvm
$ ninja check-mlir check-clang
$ ninja
$ ninja check-buddy
```

### Two-step building strategy

If you have not previously built llvm-project and you want to use LLVM/MLIR as an external library, you can follow these steps to build llvm-project first, and then build buddy-mlir.

#### Build and Test LLVM/MLIR/CL

```
$ cd buddy-mlir
$ mkdir llvm/build
$ cd llvm/build
$ cmake -G Ninja ../llvm \
    -DLLVM_ENABLE_PROJECTS="mlir;clang" \
    -DLLVM_TARGETS_TO_BUILD="host;RISCV" \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=RELEASE
$ ninja check-mlir check-clang
```

If your target machine includes a Nvidia GPU, you can use the following configuration:

```
$ cmake -G Ninja ../llvm \
    -DLLVM_ENABLE_PROJECTS="mlir;clang" \
    -DLLVM_TARGETS_TO_BUILD="host;RISCV;NVPTX" \
    -DMLIR_ENABLE_CUDA_RUNNER=ON \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=RELEASE
```

If your target machine has lld installed, you can use the following configuration:

```
$ cmake -G Ninja ../llvm \
    -DLLVM_ENABLE_PROJECTS="mlir;clang" \
    -DLLVM_TARGETS_TO_BUILD="host;RISCV" \
    -DLLVM_USE_LINKER=lld \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=RELEASE
```

#### Build buddy-mlir

If you have previously built the llvm-project, you can replace the $PWD with the path to the directory where you have successfully built the llvm-project.

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
$ ninja check-buddy
```

If you want to add domain-specific framework support, please add the following cmake options:

| Framework  | Enable Option | Other Options |
| -------------- | ------------- | ------------- |
| OpenCV  | `-DBUDDY_ENABLE_OPENCV=ON`  | Add `-DOpenCV_DIR=</PATH/TO/OPENCV/BUILD/>` or install OpenCV release version on your local device. |

## Dialects

### Bud Dialect

Bud dialect is designed for testing and demonstrating.

### DIP Dialect

DIP dialect is designed for digital image processing abstraction.

## Tools

### buddy-opt

The buddy-opt is the driver for dialects and optimization in buddy-mlir project. 

### AutoConfig Mechanism

The `AutoConfig` mechanism is designed to detect the target hardware and configure the toolchain automatically.

## Examples

The purpose of the examples is to give users a better understanding of how to use the passes and the interfaces in buddy-mlir. Currently, we provide three types of examples.

- IR level conversion and transformation examples.
- Domain-specific application level examples.
- Testing and demonstrating examples.

For more details, please see the [documentation of the examples](./examples/README.md).

## Benchmarks

The benchmarks in this repo use JIT tool (mlir-cpu-runner) as the execution engine.
For AOT benchmarks, please see [buddy-benchmark repo](https://github.com/buddy-compiler/buddy-benchmark).

We provide the following benchmarks:

- Conv2D

```
$ cd buddy-mlir/benchmark
$ make
```

For more features and configurations, please see the [benchmark document](./benchmark/README.md).
