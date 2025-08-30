module attributes {gpu.container_module} {
  gpu.module @kernels {
    gpu.func @vector_bitcast(%ret0: memref<3xi64>, %ret1: memref<6xf32>, %ret2: memref<6xi32>) kernel {
      %c0 = arith.constant 0 : index
      %v0 = arith.constant dense<[10, 20, 56, 90, 12, 90]> : vector<6xi32>
      %v1 = vector.bitcast %v0 : vector<6xi32> to vector<3xi64>
      vector.store %v1, %ret0[%c0] : memref<3xi64>, vector<3xi64>

      %v2 = vector.bitcast %v0 : vector<6xi32> to vector<6xf32>
      vector.store %v2, %ret1[%c0] : memref<6xf32>, vector<6xf32>

      %v3 = vector.bitcast %v2 : vector<6xf32> to vector<6xi32>
      vector.store %v3, %ret2[%c0] : memref<6xi32>, vector<6xi32>

      gpu.return
    }
  }

  func.func @main() {
    %c1 = arith.constant 1 : index
    %kernel_ret0 = memref.alloc() : memref<3xi64>
    %kernel_ret0_cast = memref.cast %kernel_ret0 : memref<3xi64> to memref<*xi64>

    %kernel_ret1 = memref.alloc() : memref<6xf32>
    %kernel_ret1_cast = memref.cast %kernel_ret1 : memref<6xf32> to memref<*xf32>

    %kernel_ret2 = memref.alloc() : memref<6xi32>
    %kernel_ret2_cast = memref.cast %kernel_ret2 : memref<6xi32> to memref<*xi32>

    gpu.host_register %kernel_ret0_cast : memref<*xi64>
    gpu.host_register %kernel_ret1_cast : memref<*xf32>
    gpu.host_register %kernel_ret2_cast : memref<*xi32>
    gpu.launch_func @kernels::@vector_bitcast blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) args(%kernel_ret0 : memref<3xi64>, %kernel_ret1 : memref<6xf32>, %kernel_ret2 : memref<6xi32>)

    call @printMemrefI64(%kernel_ret0_cast) : (memref<*xi64>) -> ()
    call @printMemrefF32(%kernel_ret1_cast) : (memref<*xf32>) -> ()
    call @printMemrefI32(%kernel_ret2_cast) : (memref<*xi32>) -> ()

    func.return
  }
  func.func private @printMemrefI64(%tpr : memref<*xi64>)
  func.func private @printMemrefF32(%ptr : memref<*xf32>)
  func.func private @printMemrefI32(%ptr : memref<*xi32>)

}
