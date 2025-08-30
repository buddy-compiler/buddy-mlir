module attributes {gpu.container_module} {   
  gpu.module @kernels {
    gpu.func @kernel_1 (%arg0 : f32, %arg1 : memref<2x2x2xf32>) kernel {
      %tIdX = gpu.thread_id x
      %tIdY = gpu.thread_id y
      %tIdZ = gpu.thread_id z 
      memref.store %arg0, %arg1[%tIdX, %tIdY, %tIdZ] : memref<2x2x2xf32>  
      gpu.return 
    }
  }

  func.func @main() {
    %cst1 = arith.constant 1 : index 
    %cst2 = arith.constant 2 : index
    %cst5 = arith.constant 5 : index 
    %arg0 = arith.constant 1.0 : f32
    %arg1 = memref.alloc() : memref<2x2x2xf32>
    %arg1_cast = memref.cast %arg1 : memref<2x2x2xf32> to memref<*xf32>
    gpu.host_register %arg1_cast : memref<*xf32>
    gpu.launch_func @kernels::@kernel_1 blocks in (%cst1, %cst1, %cst1) threads in (%cst2,%cst2,%cst2) args(%arg0 : f32, %arg1 : memref<2x2x2xf32>)
    call @printMemrefF32(%arg1_cast) : (memref<*xf32>) -> ()
    memref.dealloc %arg1 : memref<2x2x2xf32>
    return 
  }
  func.func private @printMemrefF32(%ptr : memref<*xf32>) 
}

