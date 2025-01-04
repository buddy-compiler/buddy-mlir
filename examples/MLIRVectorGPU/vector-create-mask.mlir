module attributes {gpu.container_module} {
  gpu.module @kernels {
    gpu.func @vector_create_mask(%result: memref<3xi1>) kernel {
      %c0 = arith.constant 0 : index
      %cons2 = arith.constant 2 : index
      %mask0 = vector.create_mask %cons2 : vector<3xi1>
      vector.store %mask0, %result[%c0] : memref<3xi1>, vector<3xi1>
      gpu.return
    }
  }
  func.func @main() {
    %result = memref.alloc() : memref<3xi1>
    %result_cast = memref.cast %result : memref<3xi1> to memref<*xi1>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    gpu.host_register %result_cast : memref<*xi1>
    gpu.launch_func @kernels::@vector_create_mask blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) args(%result : memref<3xi1>)
    %result_v = vector.load %result[%c0] : memref<3xi1>, vector<3xi1>
    vector.print %result_v : vector<3xi1>

    func.return
  }
}
