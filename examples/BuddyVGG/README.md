# Buddy Compiler VGG Example

## VGG Model Inference

0. Activate your python environment.

1. Build buddy-mlir

2. Set the `PYTHONPATH` environment variable.

Make sure you are in the build directory.

```bash
$ export BUDDY_MLIR_BUILD_DIR=$PWD
$ export LLVM_MLIR_BUILD_DIR=$PWD/../llvm/build
$ export PYTHONPATH=${LLVM_MLIR_BUILD_DIR}/tools/mlir/python_packages/mlir_core:${BUDDY_MLIR_BUILD_DIR}/python_packages:${PYTHONPATH}
```

3. Set the `VGG_EXAMPLE_PATH` environment variable.

```bash
$ export VGG_EXAMPLE_PATH=${BUDDY_MLIR_BUILD_DIR}/../examples/BuddyVGG/
```

4. Build and run the VGG example

```bash
$ cmake -G Ninja .. -DBUDDY_VGG_EXAMPLES=ON
$ ninja buddy-vgg-run
$ cd bin
$ ./buddy-vgg-run
```

## Debug the Lowering Pass Pipeline with Fake Parameters.

```bash
$ cd buddy-mlir
$ cd examples/BuddyVGG
$ make buddy-vgg-lower
$ make buddy-vgg-translate
$ make buddy-vgg-run
```
