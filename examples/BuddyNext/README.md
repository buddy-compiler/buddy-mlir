# Next Exploration Steps in Buddy MLIR

## Build llvm/mlir

```
$ cd buddy-mlir/llvm
$ mkdir build && cd build
$ cmake -G Ninja ../llvm \
    -DLLVM_ENABLE_PROJECTS="mlir;clang" \
    -DLLVM_TARGETS_TO_BUILD="host;RISCV" \
    -DLLVM_ENABLE_RUNTIMES="openmp" \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DOPENMP_ENABLE_LIBOMPTARGET=OFF \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
    -DPython3_EXECUTABLE=$(which python3)
$ ninja check-clang check-mlir
$ ninja -C runtimes/runtimes-bins omp
```
Some test errors may occur when running check-mlir on RISC-V platforms, please ignore them.


## Build buddy-mlir

Build MLIR/LLVM toolchains and libraries

```
$ cd buddy-mlir
$ mkdir build && cd build
$ cmake -G Ninja .. \
    -DMLIR_DIR=$PWD/../llvm/build/lib/cmake/mlir \
    -DLLVM_DIR=$PWD/../llvm/build/lib/cmake/llvm \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=RELEASE \
$ ninja
```

## Run the examples

```
$ cd examples/MLIRNext
$ make <target name> (e.g. `make next-mhsa-core-aot-omp`)
```
