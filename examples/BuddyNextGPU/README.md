# Next Exploration Steps in Buddy MLIR (GPU)

## Build llvm/mlir with CUDA support

```bash
$ cd buddy-mlir/llvm
$ mkdir build && cd build
$ cmake -G Ninja ../llvm \
    -DLLVM_ENABLE_PROJECTS="mlir;clang" \
    -DLLVM_TARGETS_TO_BUILD="host;NVPTX" \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DMLIR_ENABLE_CUDA_RUNNER=ON \
    -DMLIR_ENABLE_NVPTXCOMPILER=ON \
    -DMLIR_NVPTXCOMPILER_LIB_PATH=/usr/local/cuda/lib64/libnvptxcompiler_static.a
$ ninja mlir-opt mlir-runner mlir-cuda-runner
```

## Build buddy-mlir

```bash
$ cd buddy-mlir
$ mkdir build && cd build
$ cmake -G Ninja .. \
    -DMLIR_DIR=$PWD/../llvm/build/lib/cmake/mlir \
    -DLLVM_DIR=$PWD/../llvm/build/lib/cmake/llvm \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=RELEASE
$ ninja buddy-opt
```

## Run the examples

```bash
$ cd examples/BuddyNextGPU
$ make next-attention-run
```

Override GPU target (default: sm_80):

```bash
$ make next-attention-run
```

## Pipeline overview

```
TOSA → linalg (tosa-to-linalg-named/linalg/tensor/arith)
     → bufferize (one-shot-bufferize, identity-layout-map)
     → GPU mapping (linalg→parallel-loops→gpu, kernel-outlining)
     → memcpy insertion (convert-memcpy-to-gpu, gpu-async-region)
     → NVVM compilation (gpu-lower-to-nvvm-pipeline, fatbin)
     → host lowering (gpu-to-llvm, convert-func-to-llvm, ...)
     → execute (mlir-runner + libmlir_cuda_runtime.so)
```
