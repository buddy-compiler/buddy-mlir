# Buddy Compiler Fuse Ops Example

## Fused LeNet Model Inference

0. Activate your python environment.

1. Build buddy-mlir

```bash
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

2. Set the `PYTHONPATH` environment variable.

Make sure you are in the build directory.

```bash
$ export BUDDY_MLIR_BUILD_DIR=$PWD
$ export LLVM_MLIR_BUILD_DIR=$PWD/../llvm/build
$ export PYTHONPATH=${LLVM_MLIR_BUILD_DIR}/tools/mlir/python_packages/mlir_core:${BUDDY_MLIR_BUILD_DIR}/python_packages:${PYTHONPATH}
```

3. Set the `FUSED_LENET_EXAMPLE_PATH` environment variable.

```bash
$ export FUSED_LENET_EXAMPLE_PATH=${BUDDY_MLIR_BUILD_DIR}/../examples/BuddyFusedLeNet/
```

4. Build and run the Fused LeNet example

```bash
$ cmake -G Ninja .. -DBUDDY_FUSED_LENET_EXAMPLES=ON
$ ninja buddy-fused-lenet-run
$ cd bin
$ ./buddy-fused-lenet-run
```
