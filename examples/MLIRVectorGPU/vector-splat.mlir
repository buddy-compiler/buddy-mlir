module attributes {gpu.container_module} {
  gpu.module @kernels {
    gpu.func @vector_splat(%result: memref<3xf32>) kernel {
      %c0 = arith.constant 0 : index
      %c10 = arith.constant 10.0 : f32
      %v1 = vector.splat %c10 : vector<3xf32>
      vector.store %v1, %result[%c0] : memref<3xf32>, vector<3xf32>
      gpu.return
    }
  }

  func.func @main() {
    %result = memref.alloc() : memref<3xf32>
    %result_cast = memref.cast %result : memref<3xf32> to memref<*xf32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    gpu.host_register %result_cast : memref<*xf32>
    gpu.launch_func @kernels::@vector_splat blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) args(%result : memref<3xf32>)
    %result_v = vector.load %result[%c0] : memref<3xf32>, vector<3xf32>
    vector.print %result_v : vector<3xf32>
    func.return
  }
}
