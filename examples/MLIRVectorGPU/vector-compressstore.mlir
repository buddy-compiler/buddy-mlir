module attributes {gpu.container_module} {
  gpu.module @kernels {
    gpu.func @vector_compressstore(%base0 : memref<8xi32>, %base1 : memref<4x4xi32>) kernel {

      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c3 = arith.constant 3 : index

      // case 0
      %mask0 = arith.constant dense<[1, 0, 1]> : vector<3xi1>
      %value0 = arith.constant dense<[100, 101, 102]> : vector<3xi32>

      vector.compressstore %base0[%c0], %mask0, %value0 : memref<8xi32>, vector<3xi1>, vector<3xi32>

      // case 1
      %base1_casted = memref.cast %base1 : memref<4x4xi32> to memref<?x?xi32>
      %mask1 = arith.constant dense<[1, 0, 1, 1, 1, 1, 0, 0]> : vector<8xi1>
      %value1 = arith.constant dense<[500, 501, 502, 503, 504, 505, 506, 507]> : vector<8xi32>

      vector.compressstore %base1_casted[%c3, %c1], %mask1, %value1
        : memref<?x?xi32>, vector<8xi1>, vector<8xi32>

      gpu.return
    }
  } 

  memref.global "private" @gv0 : memref<8xi32> = dense<[0, 1, 2, 3, 4, 5, 6, 7]>

  memref.global "private" @gv1 : memref<4x4xi32> = dense<[[0, 1, 2, 3],
                                                          [4, 5, 6, 7],
                                                          [8, 9, 10, 11],
                                                          [12, 13, 14, 15]]>

  func.func @main() {
    %A = memref.get_global @gv0 : memref<8xi32>
    %B = memref.get_global @gv1 : memref<4x4xi32>
    %A_cast = memref.cast %A : memref<8xi32> to memref<*xi32>
    %B_cast = memref.cast %B : memref<4x4xi32> to memref<*xi32>
    %c1 = arith.constant 1 : index
    gpu.host_register %A_cast : memref<*xi32>
    gpu.host_register %B_cast : memref<*xi32>
    gpu.launch_func @kernels::@vector_compressstore blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) args(%A : memref<8xi32>, %B : memref<4x4xi32>)

    call @printMemrefI32(%A_cast) : (memref<*xi32>) -> ()
    call @printMemrefI32(%B_cast) : (memref<*xi32>) -> ()

    func.return
  }
  func.func private @printMemrefI32(%ptr : memref<*xi32>)
}
