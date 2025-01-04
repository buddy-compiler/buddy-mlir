module attributes {gpu.container_module} {
  gpu.module @kernels {
    memref.global "private" @gv0 : memref<2x4xi32> = dense<[[1, 2, 3, 4], [5, 6, 7, 8]]>
    gpu.func @vector_type_cast(%result : memref<8xi32>) kernel {
      %mem0 = memref.get_global @gv0 : memref<2x4xi32>
      %m0 = vector.type_cast %mem0 : memref<2x4xi32> to memref<vector<2x4xi32>>
      %v0 = memref.load %m0[] : memref<vector<2x4xi32>>
      %v0_reshape = vector.shape_cast %v0 : vector<2x4xi32> to vector<8xi32>
      %c0 = arith.constant 0 : index
      vector.store %v0_reshape, %result[%c0] : memref<8xi32>, vector<8xi32>
      gpu.return
    }
  }

  func.func @main() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    %result = memref.alloc() : memref<8xi32>
    %result_cast = memref.cast %result : memref<8xi32> to memref<*xi32>

    gpu.host_register %result_cast : memref<*xi32>

    gpu.launch_func @kernels::@vector_type_cast blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) args(%result : memref<8xi32>)

    %result_v = vector.load %result[%c0] : memref<8xi32>, vector<8xi32>
    %result_v_reshape = vector.shape_cast %result_v : vector<8xi32> to vector<2x4xi32>
    vector.print %result_v_reshape : vector<2x4xi32>

    func.return 
  }
}
