# Buddy Compiler DenseNet Image Classification Example

## Introduction
This example shows how to use Buddy Compiler to compile a DenseNet model to MLIR code then run it.  The [model](DenseNet121) is trained to classify the image type.


## How to run
1. Ensure that LLVM, OpenCV, Buddy Compiler and the Buddy Compiler python packages are installed properly. You can refer to [here](https://github.com/buddy-compiler/buddy-mlir) for more information and do a double check.

2. Set the `PYTHONPATH` environment variable.
```bash
$ export PYTHONPATH=/path-to-buddy-mlir/llvm/build/tools/mlir/python_packages/mlir_core:/path-to-buddy-mlir/build/python_packages:${PYTHONPATH}
```

3. Build and run the BERT example
```bash
$ cmake -G Ninja .. -DBUDDY_DENSENET_EXAMPLES=ON
$ ninja buddy-densenet-run
$ cd bin
$ ./buddy-densenet-run
```

4. Enjoy it!
