module attributes {gpu.container_module} {
  gpu.module @kernels {
    gpu.func @vector_outerproduct(%result0: memref<9xi32>, %result1: memref<3xi32>) kernel {
      %c0 = arith.constant 0 : index
      %v0 = arith.constant dense<[1, 2, 3]> : vector<3xi32>
      %v1 = arith.constant dense<[4, 5, 6]> : vector<3xi32>
      %v0xv1 = vector.outerproduct %v0, %v1 : vector<3xi32>, vector<3xi32>
      %v0xv1_shape_casted = vector.shape_cast %v0xv1 : vector<3x3xi32> to vector<9xi32>
      vector.store %v0xv1_shape_casted, %result0[%c0] : memref<9xi32>, vector<9xi32>

      %s0 = arith.constant 3 : i32
      %s0xv0 = vector.outerproduct %v0, %s0 : vector<3xi32>, i32
      %s0xv0_shape_casted = vector.shape_cast %s0xv0 : vector<3xi32> to vector<3xi32>
      vector.store %s0xv0_shape_casted, %result1[%c0] : memref<3xi32>, vector<3xi32>

      gpu.return
    }
  }

  func.func @main() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %result0 = memref.alloc() : memref<9xi32>
    %result0_cast = memref.cast %result0 : memref<9xi32> to memref<*xi32>
    %result1 = memref.alloc() : memref<3xi32>
    %result1_cast = memref.cast %result1 : memref<3xi32> to memref<*xi32>
    gpu.host_register %result0_cast : memref<*xi32>
    gpu.host_register %result1_cast : memref<*xi32>

    gpu.launch_func @kernels::@vector_outerproduct blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) args(%result0 : memref<9xi32>, %result1 : memref<3xi32>)

    %result0_v = vector.load %result0[%c0] : memref<9xi32>, vector<9xi32>
    %result0_v_reshape = vector.shape_cast %result0_v : vector<9xi32> to vector<3x3xi32>
    vector.print %result0_v_reshape : vector<3x3xi32>

    %result1_v = vector.load %result1[%c0] : memref<3xi32>, vector<3xi32>
    vector.print %result1_v : vector<3xi32> 

    func.return
  }
}
