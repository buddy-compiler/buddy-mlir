# Buddy Compiler TensorOps Example

## Environment BuildUp

1. Build buddy-mlir, make sure you enable MLIR Python bindings.

```shell
$ mkdir build && cd build
$ cmake -G Ninja .. \
    -DMLIR_DIR=$PWD/../llvm/build/lib/cmake/mlir \
    -DLLVM_DIR=$PWD/../llvm/build/lib/cmake/llvm \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DBUDDY_MLIR_ENABLE_PYTHON_PACKAGES=ON \
    -DPython3_EXECUTABLE=$(which python3) \
$ ninja
$ ninja check-buddy
```

2. Set the `PYTHONPATH` environment variable (make sure you are in `buddy-mlir/build` directory).

```shell
$ export BUDDY_MLIR_BUILD_DIR=$PWD
$ export LLVM_MLIR_BUILD_DIR=$PWD/../llvm/build
$ export PYTHONPATH=${LLVM_MLIR_BUILD_DIR}/tools/mlir/python_packages/mlir_core:${BUDDY_MLIR_BUILD_DIR}/python_packages:${PYTHONPATH}
```

3. Build the `TensorOps` example.

```shell
$ cmake -G Ninja .. -DTENSOR_OPS_EXAMPLES=ON
$ ninja tensor-ops-run
$ cd bin
$ ./tensor-ops-run
```

4. Feel free to generate your own mlir operation(e.g. `mulOp`) and run it!