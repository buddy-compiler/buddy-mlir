# Buddy Compiler TOSA ADD Operator Test (f32)

## Introduction
This example demonstrates how to use Buddy Compiler to compile a simple sample that utilizes the TOSA ADD operator (f32) to MLIR code and execute it.

## How to run
1. Ensure that LLVM, Buddy Compiler, and the Buddy Compiler Python packages are installed properly. For more information and to double-check the installation, refer to [here](https://github.com/buddy-compiler/buddy-mlir).

2. Set the `PYTHONPATH` environment variable.
```bash
$ export PYTHONPATH=/path-to-buddy-mlir/llvm/build/tools/mlir/python_packages/mlir_core:/path-to-buddy-mlir/build/python_packages:${PYTHONPATH}
```

3. Build and run the TOSA ADD example
```bash
$ cmake -G Ninja .. -DBUDDY_TOSA_EXAMPLES=ON
$ ninja buddy-add-run
$ cd bin
$ ./buddy-add-run
```
