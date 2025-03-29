# Buddy Compiler BERT Emotion Classification Example

## Introduction
This example shows how to use Buddy Compiler to compile a simple model that only use add operator to MLIR code then run it.


## How to run
1. Ensure that LLVM, Buddy Compiler and the Buddy Compiler python packages are installed properly. You can refer to [here](https://github.com/buddy-compiler/buddy-mlir) for more information and do a double check.

2. Set the `PYTHONPATH` environment variable.
```bash
$ export PYTHONPATH=/path-to-buddy-mlir/llvm/build/tools/mlir/python_packages/mlir_core:/path-to-buddy-mlir/build/python_packages:${PYTHONPATH}
```

3. Build and run the BERT example
```bash
$ cmake -G Ninja .. -DBUDDY_ADD_EXAMPLES=ON
$ ninja buddy-add-run
$ cd bin
$ ./buddy-add-run
```

4. Enjoy it!
