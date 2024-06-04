# **Buddy Compiler Quantization Example**



## Statement

This example is for testing the quantification of `mul` and `embedding` operations. Please ensure that the beginning section is completed before running [started part](https://github.com/buddy-compiler/buddy-mlir/blob/main/README.md)



## How to run

The bash command is as follows：

```bash
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
$ cmake -G Ninja .. -DBUDDY_TESTFLOAT16=ON
$ ninja buddy-testfloat16-embedding-run
$ ninja buddy-testfloat16-run
$ ./bin/buddy-testfloat16-run
$ ./bin/buddy-testfloat16-embedding-run
```


