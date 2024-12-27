module attributes {gpu.container_module} {
  gpu.module @kernels {
    gpu.func @vector_constant_mask() kernel {
      %c0 = arith.constant 0 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %mask0_vec = vector.constant_mask [3, 2] : vector<4x3xi1>
      
      %mask0_0_0 = vector.extract %mask0_vec[0, 0] : i1 from vector<4x3xi1>
      %mask0_0_1 = vector.extract %mask0_vec[0, 1] : i1 from vector<4x3xi1>
      %mask0_0_2 = vector.extract %mask0_vec[0, 2] : i1 from vector<4x3xi1>
      %mask0_1_0 = vector.extract %mask0_vec[1, 0] : i1 from vector<4x3xi1>
      %mask0_1_1 = vector.extract %mask0_vec[1, 1] : i1 from vector<4x3xi1>
      %mask0_1_2 = vector.extract %mask0_vec[1, 2] : i1 from vector<4x3xi1>
      %mask0_2_0 = vector.extract %mask0_vec[2, 0] : i1 from vector<4x3xi1>
      %mask0_2_1 = vector.extract %mask0_vec[2, 1] : i1 from vector<4x3xi1>
      %mask0_2_2 = vector.extract %mask0_vec[2, 2] : i1 from vector<4x3xi1>
      %mask0_3_0 = vector.extract %mask0_vec[3, 0] : i1 from vector<4x3xi1>
      %mask0_3_1 = vector.extract %mask0_vec[3, 1] : i1 from vector<4x3xi1>
      %mask0_3_2 = vector.extract %mask0_vec[3, 2] : i1 from vector<4x3xi1>

      gpu.printf "%d " %mask0_0_0 : i1
      gpu.printf "%d " %mask0_0_1 : i1
      gpu.printf "%d\n" %mask0_0_2 : i1
      gpu.printf "%d " %mask0_1_0 : i1
      gpu.printf "%d " %mask0_1_1 : i1
      gpu.printf "%d\n" %mask0_1_2 : i1
      gpu.printf "%d " %mask0_2_0 : i1
      gpu.printf "%d " %mask0_2_1 : i1
      gpu.printf "%d\n" %mask0_2_2 : i1
      gpu.printf "%d " %mask0_3_0 : i1
      gpu.printf "%d " %mask0_3_1 : i1
      gpu.printf "%d\n" %mask0_3_2 : i1
      // CHECK: vector.store for i1 is not supported?
      // vector.store %mask0_vec, %mask0[%c0, %c0] : memref<4x3xi1>, vector<4x3xi1>
      gpu.return
    }
  }

  func.func @main() {
    %mask_created = memref.alloc() : memref<4x3xi1>
    %mask_created_cast = memref.cast %mask_created : memref<4x3xi1> to memref<*xi1>
    %c1 = arith.constant 1 : index
    gpu.host_register %mask_created_cast : memref<*xi1>
    gpu.launch_func @kernels::@vector_constant_mask blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) args()

    func.return
  }
  func.func private @printMemrefI32(%ptr : memref<*xi1>)
}