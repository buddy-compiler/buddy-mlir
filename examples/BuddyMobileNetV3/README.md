# Buddy Compiler MobileNetV3 Example

## MobileNetV3 Model Inference

0. Activate your python environment.

1. Build buddy-mlir

```bash
$ cd buddy-mlir
$ mkdir build && cd build
$ cmake -G Ninja .. \
    -DMLIR_DIR=$PWD/../llvm/build/lib/cmake/mlir \
    -DLLVM_DIR=$PWD/../llvm/build/lib/cmake/llvm \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DBUDDY_MLIR_ENABLE_PYTHON_PACKAGES=ON \
    -DPython3_EXECUTABLE=$(which python3) \
    -DBUDDY_ENABLE_OPENCV=ON \
    -DOpenCV_DIR=</PATH/TO/OPENCV/BUILD/>
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

3. Set the `MOBILENETV3_EXAMPLE_PATH` environment variable.

```bash
$ export MOBILENETV3_EXAMPLE_PATH=${BUDDY_MLIR_BUILD_DIR}/../examples/BuddyMobileNetV3/
```

4. Build and run the MobileNetV3 example

```bash
$ cmake -G Ninja .. -DBUDDY_MOBILENETV3_EXAMPLES=ON
$ ninja buddy-mobilenetv3-run
$ cd bin
$ ./buddy-mobilenetv3-run
```

