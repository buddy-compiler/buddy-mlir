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
    -DBUDDY_MLIR_ENABLE_DIP_LIB=ON
$ ninja
$ ninja check-buddy
```

2. Same config as your example

3. Set BUDDY_PROFILE_EXAMPLES=ON

4. Build profiler
```bash
$ cmake -G Ninja .. -DBUDDY_PROFILE_EXAMPLES=ON
$ ninja
```

5. Set the `your example path` environment variable.

```bash
$ export MOBILENETV3_EXAMPLE_PATH=$PWD
```

```bash
$ ./buddy-mobilenetv3-profiling
```

