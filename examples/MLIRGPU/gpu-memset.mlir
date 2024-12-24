// RUN: mlir-opt gpu-memset.mlir -gpu-kernel-outlining | \
// RUN: mlir-opt -pass-pipeline="builtin.module(nvvm-attach-target,\
// RUN: gpu.module(convert-gpu-to-nvvm), gpu-to-llvm, gpu-module-to-binary)" |\
// RUN: mlir-cpu-runner -entry-point-result=void -shared-libs=${MLIR_CUDA_RUNTIME} -shared-libs=${MLIR_RUNNER_UTILS}

// The gpu.memset operation is a memory initialization operation that sets the content of 
// a memref to a scalar value on the GPU device
func.func @main() {
  %data = memref.alloc() : memref<1x6xi32>
  %cst0 = arith.constant 0 : i32
  %cst1 = arith.constant 1 : i32
  %cst2 = arith.constant 2 : i32
  %cst4 = arith.constant 4 : i32
  %cst8 = arith.constant 8 : i32
  %cst16 = arith.constant 16 : i32

  %value = arith.constant 0 : i32

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c5 = arith.constant 5 : index
  %c6 = arith.constant 6 : index

  memref.store %cst0, %data[%c0, %c0] : memref<1x6xi32>
  memref.store %cst1, %data[%c0, %c1] : memref<1x6xi32>
  memref.store %cst2, %data[%c0, %c2] : memref<1x6xi32>
  memref.store %cst4, %data[%c0, %c3] : memref<1x6xi32>
  memref.store %cst8, %data[%c0, %c4] : memref<1x6xi32>
  memref.store %cst16, %data[%c0, %c5] : memref<1x6xi32>
  
  %cast_data = memref.cast %data : memref<1x6xi32> to memref<*xi32>
  gpu.host_register %cast_data : memref<*xi32>

  %t0 = gpu.wait async
  // perform a memset operation on the memref %data with the value 0 using gpu.memset async, 
  // which is an asynchronous memory initialization operation. The operation takes a token %t0 as 
  // the asynchronous dependency, which represents the completion of the previous gpu.wait async operation. 
  // The operation returns a new token %t1, which represents the completion of the gpu.memset async operation.
  %t1 = gpu.memset async [%t0] %data, %value : memref<1x6xi32>, i32 

  %cast_memset_data = memref.cast %data : memref<1x6xi32> to memref<*xi32>
  gpu.host_register %cast_memset_data : memref<*xi32>
  // CHECK: [[0,   0,   0,   0,   0,   0]]
  call @printMemrefI32(%cast_memset_data) : (memref<*xi32>) -> ()
  return
}

func.func private @printMemrefI32(memref<*xi32>)