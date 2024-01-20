// RUN: mlir-opt %s -gpu-kernel-outlining \
// RUN: | mlir-opt -pass-pipeline="builtin.module(nvvm-attach-target{chip=sm_70 O=3},\
// RUN:     gpu.module(convert-gpu-to-nvvm), gpu-to-llvm, gpu-module-to-binary)" \
// RUN: | mlir-cpu-runner -entry-point-result=void -shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_CUDA_RUNTIME}


func.func @main() {
  %arg = memref.alloc() : memref<13xf32>
  %dst = memref.cast %arg : memref<13xf32> to memref<?xf32>
  %one = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %sx = memref.dim %dst, %c0 : memref<?xf32>
  %cast_dst = memref.cast %dst : memref<?xf32> to memref<*xf32>
  gpu.host_register %cast_dst : memref<*xf32>
  // launch a gpu kernel
  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %one, %grid_y = %one, %grid_z = %one)
             threads(%tx, %ty, %tz) in (%block_x = %sx, %block_y = %one, %block_z = %one) {
    // convert index to i32 and bind %tx to %t0
    %t0 = arith.index_cast %tx : index to i32
    // convert i32 to f32
    %val = arith.sitofp %t0 : i32 to f32
    %width = arith.index_cast %block_x : index to i32
    %offset = arith.constant 4 : i32
    // Based on the bool value of %valid, choose to jump to the two basic blocks ^bb1 or ^bb0 
    // and pass the value of %shfl as the parameter
    // The shuffle operation uses the xor mode, which means that each thread exchanges its value with 
    // another thread whose index is obtained by bitwise xor with a given offset. The offset is 4, 
    // so the threads with indices 0, 4, 8, and 12 swap their values with the threads with indices 4, 0, 12, 
    // and 8, respectively. The threads with indices 1, 5, 9, and 13 swap their values with the threads with indices 
    // 5, 1, 13, and 9, respectively. The threads with indices 2, 6, 10, and 14 swap their values with the threads 
    // with indices 6, 2, 14, and 10, respectively. The threads with indices 3, 7, 11, and 15 swap their values 
    // with the threads with indices 7, 3, 15, and 11, respectively. If the thread index is out of bounds, 
    // the shuffle operation returns a false value for the validity flag. In that case, the thread stores -1.0 to 
    // the buffer. Otherwise, the thread stores the shuffled value to the buffer. 
    // The initial values of the buffer are the thread indices converted to floating-point values, 
    // i.e., [0.0, 1.0, 2.0, …, 12.0]. After the shuffle operation, 
    // the values are [4.0, 5.0, 6.0, 7.0, 0.0, 1.0, 2.0, 3.0, 12.0, -1.0, -1.0, -1.0, 8.0]. 
    %shfl, %valid = gpu.shuffle xor %val, %offset, %width : f32
    cf.cond_br %valid, ^bb1(%shfl : f32), ^bb0
  ^bb0:
  // if the target thread is invalid, jump to the ^bb0 block
  // set %m1 = -1.0 and jump to ^bb1 and pass the %m1 as the parameter
    %m1 = arith.constant -1.0 : f32
    cf.br ^bb1(%m1 : f32)
  ^bb1(%value : f32):
    memref.store %value, %dst[%tx] : memref<?xf32>
    gpu.terminator
  }
  // CHECK: [4, 5, 6, 7, 0, 1, 2, 3, 12, -1, -1, -1, 8]
  call @printMemrefF32(%cast_dst) : (memref<*xf32>) -> ()
  return
}

func.func private @printMemrefF32(%ptr : memref<*xf32>)
