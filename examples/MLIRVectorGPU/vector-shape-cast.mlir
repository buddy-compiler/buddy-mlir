module attributes {gpu.container_module} {
  gpu.module @kernels {
    gpu.func @vector_shape_cast(%result: memref<6xi32>) kernel {
      %v0 = arith.constant dense<[[1, 2, 3], [4, 5, 6]]> 
        : vector<2x3xi32>
      %v1 = vector.shape_cast %v0 : vector<2x3xi32> to vector<6xi32>
      %c0 = arith.constant 0 : index
      vector.store %v1, %result[%c0] : memref<6xi32>, vector<6xi32>
      gpu.return
    }
  }

  func.func @main() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %result = memref.alloc() : memref<6xi32>
    %result_cast = memref.cast %result : memref<6xi32> to memref<*xi32>

    gpu.host_register %result_cast : memref<*xi32>
    gpu.launch_func @kernels::@vector_shape_cast blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) args(%result : memref<6xi32>)

    %result_v = vector.load %result[%c0] : memref<6xi32>, vector<6xi32>
    vector.print %result_v : vector<6xi32>
    func.return
  }
}
