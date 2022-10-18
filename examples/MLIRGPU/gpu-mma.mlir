module attributes {gpu.container_module} {
  func.func @main() {
    %c0 = arith.constant 0 : index 
    %c1 = arith.constant 1 : index  
    %c32 = arith.constant 32 : index 
    %c16 = arith.constant 16 : index 
    %f0 = arith.constant 0.0 : f16
    %f1 = arith.constant 1.0 : f16
    %f2 = arith.constant 2.0 : f16
    %f2f32 = arith.constant 2.0 : f32
    %f0f32 = arith.constant 0.0 : f32
    %input0 = memref.alloc() : memref<16x16xf16>
    %input1 = memref.alloc() : memref<16x16xf16>
    %output0 = memref.alloc() : memref<16x16xf32>
    %input_cast0 = memref.cast %input0 : memref<16x16xf16> to memref<*xf16>
    %input_cast1 = memref.cast %input1 : memref<16x16xf16> to memref<*xf16>
    %output_cast0 = memref.cast %output0 : memref<16x16xf32> to memref<*xf32>
    scf.for %i = %c0 to %c16 step %c1 {
      scf.for %j = %c0 to %c16 step %c1 {
        memref.store %f2, %input0[%i, %j] : memref<16x16xf16> 
        memref.store %f2, %input1[%i, %j] : memref<16x16xf16>
        memref.store %f0f32, %output0[%i, %j] : memref<16x16xf32>
      }
    }   
    call @printMemrefF32(%output_cast0) : (memref<*xf32>) -> ()
    gpu.host_register %input_cast0 : memref<*xf16>  
    gpu.host_register %input_cast1 : memref<*xf16>
    gpu.host_register %output_cast0 : memref<*xf32>
    gpu.launch_func @kernels::@kernel1 blocks in (%c1, %c1, %c1) threads in (%c32, %c1, %c1) args(%input0 : memref<16x16xf16>, %input1 : memref<16x16xf16>, %output0 : memref<16x16xf32>)
    call @printMemrefF32(%output_cast0) : (memref<*xf32>) -> ()
    memref.dealloc %input0 : memref<16x16xf16>
    memref.dealloc %input1 : memref<16x16xf16>
    memref.dealloc %output0 : memref<16x16xf32>
    return
  }

  gpu.module @kernels {
    gpu.func @kernel1(%arg0 : memref<16x16xf16>, %arg1 : memref<16x16xf16>, %arg2 : memref<16x16xf32>) kernel {
      %c0 = arith.constant 0 : index 
      %A = gpu.subgroup_mma_load_matrix %arg0[%c0, %c0] {leadDimension = 16 : index} : memref<16x16xf16> -> !gpu.mma_matrix<16x16xf16, "AOp">
      %B = gpu.subgroup_mma_load_matrix %arg1[%c0, %c0] {leadDimension = 16 : index} : memref<16x16xf16> -> !gpu.mma_matrix<16x16xf16, "BOp">
      %C = gpu.subgroup_mma_load_matrix %arg2[%c0, %c0] {leadDimension = 16 : index} : memref<16x16xf32> -> !gpu.mma_matrix<16x16xf32, "COp">
      %D = gpu.subgroup_mma_compute %A, %B, %C : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">
      gpu.subgroup_mma_store_matrix %D, %arg2[%c0, %c0] {leadDimension = 16 : index} : !gpu.mma_matrix<16x16xf32, "COp">, memref<16x16xf32>
      gpu.return
    }
  }
  func.func private @printMemrefF32(%ptr : memref<*xf32>)
}

