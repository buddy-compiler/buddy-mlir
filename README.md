# BUDDY MLIR

MLIR-Based Ideas Landing Project ([Project page](https://buddy-compiler.github.io/)).

## Getting Started

### LLVM/MLIR Dependencies

This project uses LLVM/MLIR as an external library. Please make sure [the dependencies](https://llvm.org/docs/GettingStarted.html#requirements) are available
on your machine.

### Clone and Initialize


```
$ git clone git@github.com:buddy-compiler/buddy-mlir.git
$ cd buddy-mlir
$ git submodule update --init
```

### Build and Test LLVM/MLIR/CLANG

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
    -DLLVM_ENABLE_PROJECTS="mlir" \
    -DLLVM_TARGETS_TO_BUILD="host;NVPTX" \
    -DMLIR_ENABLE_CUDA_RUNNER=ON \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=RELEASE
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
