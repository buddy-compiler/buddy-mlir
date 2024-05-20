# Buddy Compiler VGG Example

## Introduction
This example shows how to use Buddy Compiler to compile a VGG model to MLIR code then run it.  T
## How to run
1. Ensure that LLVM, Buddy Compiler and the Buddy Compiler python packages are installed properly. You can refer to [here](https://github.com/buddy-compiler/buddy-mlir) for more information and do a double check.(Note: 1.You should build llvm and buddy-mlir with python binding choice, and you'd better create a virtual env for that. 2. In your env, version of python is required below 3.12 because it seem that torch.compile() only support python<=3.11 now)

2. Set the `PYTHONPATH` environment variable.
```bash
$ export PYTHONPATH=/path-to-buddy-mlir/llvm/build/tools/mlir/python_packages/mlir_core:/path-to-buddy-mlir/build/python_packages:${PYTHONPATH}
```

3. Activate your python environment.

```bash
$ conda activate your-env
```

4. Build and run the VGG example
```bash
$ cd buddy-mlir/build
$ cmake -G Ninja .. -DBUDDY_VGG_EXAMPLES=ON
$ ninja buddy-vgg-run
$ cd bin
$ ./buddy-vgg-run
```

5. Enjoy it!
