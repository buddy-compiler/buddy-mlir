# MLIR GPU Compilation Path

## Build buddy-mlir and Run Examples

Build MLIR/LLVM toolchains and libraries

```
$ cd llvm
$ mkdir build-gpu && cd build-gpu
$ cmake -G Ninja ../llvm \
    -DLLVM_ENABLE_PROJECTS="mlir;clang" \
    -DLLVM_TARGETS_TO_BUILD="host;RISCV;NVPTX" \
    -DMLIR_ENABLE_CUDA_RUNNER=ON \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=RELEASE
$ ninja check-mlir check-clang
```

Run the exmaples

```
$ cd examples/MLIRGPU
$ make <target name> (e.g. `make gpu-all-reduce-and-jit`)
```
