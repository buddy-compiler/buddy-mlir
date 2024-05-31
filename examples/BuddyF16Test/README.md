# Buddy Compiler F16Test Example

The purpose of this example is to test the FP16&BP16 data type support.
In detail, we construct a naive torch computation graph:
```python
def squared_sum(x, y):
    t1 = (x * x).to(torch.float32)
    t2 = (y * y).to(torch.float32)
    return t1 + t2
```
where the input `x` and `y` are 5-length 1D tensor in dtype `torch.bfloat16` and `torch.float16`, respectively.

The corresponding mlir code is straight forward:
```mlir
module {
  func.func @forward(%arg0: tensor<5xbf16>, %arg1: tensor<5xf16>) -> tensor<5xf32> {
    %0 = tosa.mul %arg0, %arg0 {shift = 0 : i8} : (tensor<5xbf16>, tensor<5xbf16>) -> tensor<5xbf16>
    %1 = tosa.cast %0 : (tensor<5xbf16>) -> tensor<5xf32>
    %2 = tosa.mul %arg1, %arg1 {shift = 0 : i8} : (tensor<5xf16>, tensor<5xf16>) -> tensor<5xf16>
    %3 = tosa.cast %2 : (tensor<5xf16>) -> tensor<5xf32>
    %4 = tosa.add %1, %3 : (tensor<5xf32>, tensor<5xf32>) -> tensor<5xf32>
    return %4 : tensor<5xf32>
  }
}
```

Then we construct a use case in C++ where we put in a BFloat16 vector and a Float16 vector, and check if the output results are correct.

## Usage

1. Build and check LLVM/MLIR

```
$ cd buddy-mlir
$ mkdir llvm/build
$ cd llvm/build
$ cmake -G Ninja ../llvm \
    -DLLVM_ENABLE_PROJECTS="mlir;clang" \
    -DLLVM_TARGETS_TO_BUILD="host;RISCV" \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
    -DPython3_EXECUTABLE=$(which python3)
$ ninja check-clang check-mlir
```

2. Build and check buddy-mlir

```
$ cd buddy-mlir
$ mkdir build
$ cd build
$ cmake -G Ninja .. \
    -DMLIR_DIR=$PWD/../llvm/build/lib/cmake/mlir \
    -DLLVM_DIR=$PWD/../llvm/build/lib/cmake/llvm \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DBUDDY_MLIR_ENABLE_PYTHON_PACKAGES=ON \
    -DPython3_EXECUTABLE=$(which python3)
$ ninja
$ ninja check-buddy
```

3. Set the `PYTHONPATH` environment variable.

Make sure that the `PYTHONPATH` variable includes the directory of LLVM/MLIR python bindings and the directory of Buddy MLIR python packages.

```
$ export PYTHONPATH=/path-to-buddy-mlir/llvm/build/tools/mlir/python_packages/mlir_core:/path-to-buddy-mlir/build/python_packages:${PYTHONPATH}

// For example:
// Navigate to your buddy-mlir/build directory
$ cd buddy-mlir/build
$ export BUDDY_MLIR_BUILD_DIR=$PWD
$ export LLVM_MLIR_BUILD_DIR=$PWD/../llvm/build
$ export PYTHONPATH=${LLVM_MLIR_BUILD_DIR}/tools/mlir/python_packages/mlir_core:${BUDDY_MLIR_BUILD_DIR}/python_packages:${PYTHONPATH}
```

4. Build and run F16Test example

```
$ cmake -G Ninja .. -DBUDDY_F16TEST_EXAMPLES=ON
$ ninja buddy-f16test-run
$ ./bin/buddy-f16test-run
// expected output
Input 1: 3 3 3 3 3
Input 2: 4 4 4 4 4
Perform squared sum
Output: 25 25 25 25 25
```
