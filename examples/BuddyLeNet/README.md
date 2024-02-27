# Buddy Compiler LeNet Example

## Train the LeNet Model

Activate your python environment.

```bash
$ cd buddy-mlir
$ cd examples/BuddyLeNet
$ python pytorch-lenet-train.py
```

## LeNet Model Inference

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

3. Set the `LENET_EXAMPLE_PATH` environment variable.

```bash
$ export LENET_EXAMPLE_PATH=${BUDDY_MLIR_BUILD_DIR}/../examples/BuddyLeNet/
```

4. Build and run the LeNet example

```bash
$ cmake -G Ninja .. -DBUDDY_LENET_EXAMPLES=ON
$ ninja buddy-lenet-run
$ cd bin
$ ./buddy-lenet-run
```

## Debug the Lowering Pass Pipeline with Fake Parameters.

```bash
$ cd buddy-mlir
$ cd examples/BuddyLeNet
$ make buddy-lenet-lower
$ make buddy-lenet-translate
$ make buddy-lenet-run
```
