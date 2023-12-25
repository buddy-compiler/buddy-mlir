# Buddy Compiler LLaMA on GPU Example

** This is a work in progress. Current version of Buddy-MLIR is using an older version of LLVM, which is not compatible with the latest version of CUDA. We are working on updating the LLVM version. **

## 1. Prerequisites
Please refer to [readme-cpu.md](readme-cpu.md) for most of the steps, except for the following steps.

1. Install CUDA-toolkit
Please refer to [CUDA-toolkit](https://developer.nvidia.com/cuda-toolkit) for installation.
It is suggested that you install nsight system and nsight as well compute for profiling. Please refer to [nsight-system](https://developer.nvidia.com/nsight-systems) and [nsight-compute](https://developer.nvidia.com/nsight-compute) for installation.
Don't forget to add CUDA and other tools to your PATH.

...

For Step 4. Build and check LLVM/MLIR, please enable CUDA runner for MLIR.

```
$ cd buddy-mlir
$ mkdir llvm/build
$ cd llvm/build
$ cmake -G Ninja ../llvm \
    -DLLVM_ENABLE_PROJECTS="mlir;clang;openmp" \
    -DLLVM_TARGETS_TO_BUILD="host;RISCV;NVPTX" \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DOPENMP_ENABLE_LIBOMPTARGET=OFF \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
    -DPython3_EXECUTABLE=$(which python3) \
    -DMLIR_ENABLE_CUDA_RUNNER=ON \
    -DLLVM_CCACHE_BUILD=ON
$ ninja check-clang check-mlir omp
```

## 2. Lowering LLaMA MLIR to CUDA
We would use multiple steps to demonstrate the lowering process. Notice the first process would require the `mlir-opt` built in previous steps, but the remaining ones would need the latest version of `mlir-opt` and other llvm tools such as `llc`.

### 2.1 Lowering TOSA to Linalg
Due to the availbilty of certain operations such as `transpose`, current LLaMA lowering process would require the use of TOSA dialect. We would first lower the LLaMA model to a mixture of TOSA and Linalg dialects.
```
mlir-opt llama.mlir -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" -o llama-linalg-default.mlir
```
** Use the old version of `mlir-opt` built in previous steps. Or you might get following error: **
```
llama.mlir:747:11: error: 'tosa.mul' op attribute 'shift' failed to satisfy constraint: 8-bit signless integer attribute
    %36 = "tosa.mul"(%5, %35) {shift = 0 : i32} : (tensor<1x80x4096xf32>, tensor<1x80x1xf32>) -> tensor<1x80x4096xf32>
          ^
```
There should be no `tosa` operations in the output. Most of the operations should be `linalg` operations such as `matmul`, `batch_matmul` or `generic`.

### 2.2 Bufferizing Linalg
This step bufferizes the Linalg operations. It would fully convert the linalg-on-tensor operations to scf-on-memref operations.

- Bufferize using the old bufferization pipeline:
```
mlir-opt llama-linalg-default.mlir -arith-expand -eliminate-empty-tensors -empty-tensor-to-alloc-tensor -linalg-bufferize -convert-linalg-to-affine-loops -affine-loop-fusion -affine-parallelize -lower-affine -canonicalize -func-bufferize -arith-bufferize -tensor-bufferize -buffer-deallocation -finalizing-bufferize -canonicalize -o llama-bufferized.mlir
```

- Bufferize everything using one-shot-bufferize:
```
mlir-opt llama-linalg-default.mlir -arith-expand -eliminate-empty-tensors -empty-tensor-to-alloc-tensor -one-shot-bufferize="bufferize-function-boundaries" -expand-realloc  -resolve-shaped-type-result-dims -canonicalize -buffer-deallocation-simplification -bufferization-lower-deallocations -cse -canonicalize -buffer-deallocation-pipeline  -o llama-bufferized.mlir
```

- Bufferize everything but function boundaries using one-shot-bufferize:
```
mlir-opt llama-linalg-default.mlir -arith-expand -eliminate-empty-tensors -empty-tensor-to-alloc-tensor -one-shot-bufferize -func-bufferize -expand-realloc  -resolve-shaped-type-result-dims -canonicalize -buffer-deallocation-simplification -bufferization-lower-deallocations -finalizing-bufferize -cse -canonicalize -buffer-deallocation-pipeline  -o llama-bufferized.mlir
```

- Bufferize GPU first
```
buddy-opt -gpu-bufferize llama-linalg-default.mlir -o llama-gpu-bufferized.mlir  
```

- Bufferize everything else using one-shot-bufferize:
```
mlir-opt llama-gpu-bufferized.mlir -arith-expand -eliminate-empty-tensors -empty-tensor-to-alloc-tensor -one-shot-bufferize="bufferize-function-boundaries" -expand-realloc  -resolve-shaped-type-result-dims -canonicalize -buffer-deallocation-simplification -bufferization-lower-deallocations -cse -canonicalize -buffer-deallocation-pipeline  -o llama-bufferized.mlir
```

You should not be seeing any tensor on linalg operations. All operations would look like this:

```
scf.parallel (%arg2, %arg3) = (%c0, %c0) to (%c80, %c80) step (%c1, %c1) {
      %6 = memref.load %0[%arg3] : memref<80xi64>
      %7 = memref.load %expand_shape_650[%arg2, %c0] : memref<80x1xi64>
      %8 = arith.cmpi slt, %6, %7 : i64
      memref.store %8, %alloc_651[%arg2, %arg3] : memref<80x80xi1>
      scf.yield
    }
```

### 2.3 Converting to GPU
This step converts the scf-on-memref operations to gpu operations, with gpu kernels outlined.

```
mlir-opt llama-bufferized.mlir -gpu-map-parallel-loops -convert-parallel-loops-to-gpu -canonicalize -gpu-kernel-outlining -o llama-outlined.mlir
```

GPU kernels will be converted into separate modules and functions as such:
```
gpu.module @forward_kernel_753 {
    gpu.func @forward_kernel(%arg0: memref<80x4096xf32>, %arg1: memref<4096x4096xf32>, %arg2: memref<80x4096xf32>, %arg3: index, %arg4: index, %arg5: index) kernel attributes {gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 80, 4096, 1>} {
      %0 = gpu.block_id  x
      %1 = gpu.block_id  y
      %2 = gpu.block_id  z
      %3 = gpu.thread_id  x
      %4 = gpu.thread_id  y
      %5 = gpu.thread_id  z
      %6 = gpu.grid_dim  x
      %7 = gpu.grid_dim  y
      %8 = gpu.grid_dim  z
      %9 = gpu.block_dim  x
      %10 = gpu.block_dim  y
      %11 = gpu.block_dim  z
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      scf.for %arg6 = %arg3 to %arg4 step %arg5 {
        %12 = memref.load %arg0[%0, %arg6] : memref<80x4096xf32>
        %13 = memref.load %arg1[%arg6, %1] : memref<4096x4096xf32>
        %14 = memref.load %arg2[%0, %1] : memref<80x4096xf32>
        %15 = arith.mulf %12, %13 : f32
        %16 = arith.addf %14, %15 : f32
        memref.store %16, %arg2[%0, %1] : memref<80x4096xf32>
      }
      gpu.return
    }
  }
```

### 2.4 Converting to LLVM and NVVM operations
This step converts the operations to LLVM dialect operations, and then convert some math functions to NVVM intrinsics.

```
buddy-opt llama-outlined.mlir -gpu-host-register -o llama-host-registered.mlir
mlir-opt llama-host-registered.mlir -convert-scf-to-cf -memref-expand -finalize-memref-to-llvm -convert-arith-to-llvm -convert-gpu-to-nvvm='has-redux=1' -o llama-nvvm.mlir
```

Why do we need the `convert-gpu-to-nvvm` step? If it is not applied, and we are using the unmodified lowering pipeline from torch to linalg, the generated LLVM IR would look like this:
```
%24 = llvm.getelementptr %17[%23] : (!llvm.ptr, i64) -> !llvm.ptr, f32
%25 = llvm.load %24 : !llvm.ptr -> f32
%26 = math.fpowi %25, %arg2 : f32, i32
%27 = llvm.extractvalue %2[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
%28 = llvm.mlir.constant(327680 : index) : i64
```
For CPU, math operations such as `math.fpowi` would be lowered to LLVM intrinsics such as `llvm.powi.f32`. However, for GPU, we need to use NVVM intrinsics. And sadly there is no NVVM intrinsics for `math.fpowi`. So we would need to change the lowering pipeline to use `mlir.powf` instead. Before lowering to nvvm, it would look like this:
```
%24 = llvm.getelementptr %17[%23] : (!llvm.ptr, i64) -> !llvm.ptr, f32
%25 = llvm.load %24 : !llvm.ptr -> f32
%26 = math.powf %25, %arg2 : f32
%27 = llvm.extractvalue %2[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
%28 = llvm.mlir.constant(327680 : index) : i64
```

And after the lowering:
```
llvm.func @__nv_powf(f32, f32) -> f32
...
%41 = llvm.getelementptr %36[%40] : (!llvm.ptr, i64) -> !llvm.ptr, f32
%42 = llvm.load %41 : !llvm.ptr -> f32
%43 = llvm.call @__nv_powf(%42, %arg10) : (f32, f32) -> f32
%44 = llvm.extractvalue %27[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
```

It now uses `llvm.call` to call the NVVM intrinsics.

### 2.5 Request C wrappers
Notice that you must request wrappers before compiling GPU codes.
```
mlir-opt llama-nvvm.mlir -llvm-request-c-wrappers -o llama-wrapper.mlir
```

### 2.6 Lowering to LLVM Dialect + GPU Binary
```
mlir-opt llama-wrapper.mlir --test-lower-to-nvvm="cubin-chip=sm_80 cubin-features=+ptx71 cubin-format=fatbin" -o llama-cubin.mlir
```
Now you could use a builtin pipeline to lower code to nvvm. Notice that you must specify the chip and features. You could find the chip and features from Nvidia.
After the process all gpu code would be compiled.
```
  gpu.binary @forward_kernel_1078  [#gpu.object<#nvvm.target<chip = "sm_80", features = "+ptx71">, "...">]
```

### 2.7 Translate to LLVM IR
```
mlir-translate llama-cubin.mlir --mlir-to-llvmir -o llama.ll
```

### 2.8 Compile the LLVM IR
** Remember to use the latest version of LLC, as the latest version of MLIR generates some new intrinsics that are not supported by the old version of LLC. **
```
llc llama.ll -filetype=obj -relocation-model=pic -O3 -o llama.o
```

### 2.9 Link the object file and run
Following is an example of linking the object file with the runtime library and run the program. You could find the runtime library in the build directory of llvm-project.
```
clang llama.o llama-main.cpp.o /path-to/llvm-project/build/lib/libmlir_cuda_runtime.so /path-to/llvm-project/build/lib/libmlir_c_runner_utils.so
```
** Notice that current version of the llvm-project used by Buddy-MLIR would encounter problems with CUDA_RUNNERS enabled. Please use the latest version of MLIR for this step. **