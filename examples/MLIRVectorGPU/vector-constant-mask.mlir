module attributes {gpu.container_module} {
  gpu.module @kernels {
    gpu.func @vector_constant_mask(%result: memref<12xi1>) kernel {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %mask0_vec = vector.constant_mask [3, 2] : vector<4x3xi1>

      %mask0_shape_casted = vector.shape_cast %mask0_vec : vector<4x3xi1> to vector<12xi1>
      
      vector.store %mask0_shape_casted, %result[%c0] : memref<12xi1>, vector<12xi1>
      gpu.return
    }
  }

  func.func @main() {
    %mask_created = memref.alloc() : memref<12xi1>
    %mask_created_cast = memref.cast %mask_created : memref<12xi1> to memref<*xi1>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    gpu.host_register %mask_created_cast : memref<*xi1>
    gpu.launch_func @kernels::@vector_constant_mask blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) args(%mask_created : memref<12xi1>)
    %mask_created_vec = vector.load %mask_created[%c0] : memref<12xi1>, vector<12xi1>
    %mask_created_vec_reshape = vector.shape_cast %mask_created_vec : vector<12xi1> to vector<4x3xi1>
    vector.print %mask_created_vec_reshape : vector<4x3xi1>

    func.return
  }
  func.func private @printMemrefI32(%ptr : memref<*xi1>)
}
