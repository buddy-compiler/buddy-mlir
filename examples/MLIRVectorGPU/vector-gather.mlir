module attributes {gpu.container_module} {
  gpu.module @kernels {
    memref.global "private" @gv0 : memref<8xi32> = dense<[0, 1, 2, 3, 4, 5, 6, 7]>

    memref.global "private" @gv1 : memref<4x4xi32> = dense<[[0, 1, 2, 3],
                                                            [4, 5, 6, 7],
                                                            [8, 9, 10, 11],
                                                            [12, 13, 14, 15]]>

    memref.global "private" @gv2 : memref<8xi32> = dense<[0, 1, 2, 3, 4, 5, 6, 7]>
    gpu.func @vector_gather(%result0: memref<4xi32>, %result1: memref<4xi32>, %result2: memref<4xi32>, %result3: memref<4xi32>) kernel {

      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c3 = arith.constant 3 : index

      %base0 = memref.get_global @gv0 : memref<8xi32>
      %base1 = memref.get_global @gv1 : memref<4x4xi32>
      %base2 = memref.get_global @gv2 : memref<8xi32>

      %pass_thru_4 = arith.constant dense<[2330, 2331, 2332, 2333]> : vector<4xi32>
      %pass_thru_8 = arith.constant dense<[2330, 2331, 2332, 2333, 2334, 2335, 2336, 2337]> : vector<8xi32>
      %pass_thru_2x2 = arith.constant dense<114> : vector<2x2xi32>

      // normal
      %mask0 = arith.constant dense<1> : vector<4xi1>
      %index0 = arith.constant dense<[3, 4, 2, 1]> : vector<4xi32>
      %v0 = vector.gather %base0[%c0][%index0], %mask0, %pass_thru_4
        : memref<8xi32>, vector<4xi32>, vector<4xi1>, vector<4xi32> into vector<4xi32>
      vector.store %v0, %result0[%c0] : memref<4xi32>, vector<4xi32>

      // with mask
      %mask1 = arith.constant dense<[1, 0, 1, 0]> : vector<4xi1>
      %index1 = arith.constant dense<[3, 4, 2, 1]> : vector<4xi32>
      %v1 = vector.gather %base0[%c0][%index1], %mask1, %pass_thru_4
        : memref<8xi32>, vector<4xi32>, vector<4xi1>, vector<4xi32> into vector<4xi32>
      vector.store %v1, %result1[%c0] : memref<4xi32>, vector<4xi32>

      %mask2 = arith.constant dense<1> : vector<2x2xi1>
      %index2 = arith.constant dense<[[1, 0], [3, 2]]> : vector<2x2xi32>
      %v2 = vector.gather %base1[%c1, %c1][%index2], %mask2, %pass_thru_2x2
        : memref<4x4xi32>, vector<2x2xi32>, vector<2x2xi1>, vector<2x2xi32> into vector<2x2xi32>
      %v2_shape_casted = vector.shape_cast %v2 : vector<2x2xi32> to vector<4xi32>
      vector.store %v2_shape_casted, %result2[%c0] : memref<4xi32>, vector<4xi32>

      %mask3 = arith.constant dense<1> : vector<2x2xi1>
      %index3 = arith.constant dense<[[-1, -8], [5, 13]]> : vector<2x2xi32>
      %v3 = vector.gather %base1[%c1, %c1][%index3], %mask3, %pass_thru_2x2
        : memref<4x4xi32>, vector<2x2xi32>, vector<2x2xi1>, vector<2x2xi32> into vector<2x2xi32>
      
      // ( ( 4, 0), ( 10, 0 ) ).
      // On GPU, if indices are out-of-bound, the elements will be 0, which is different
      // from the CPU case.
      %v3_shape_casted = vector.shape_cast %v3 : vector<2x2xi32> to vector<4xi32>
      vector.store %v3_shape_casted, %result3[%c0] : memref<4xi32>, vector<4xi32>

      gpu.return
    }
  }

  func.func @main() {
    %result0 = memref.alloc() : memref<4xi32>
    %result1 = memref.alloc() : memref<4xi32>
    %result2 = memref.alloc() : memref<4xi32>
    %result3 = memref.alloc() : memref<4xi32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    // register host memory
    %result0_cast = memref.cast %result0 : memref<4xi32> to memref<*xi32>
    %result1_cast = memref.cast %result1 : memref<4xi32> to memref<*xi32>
    %result2_cast = memref.cast %result2 : memref<4xi32> to memref<*xi32>
    %result3_cast = memref.cast %result3 : memref<4xi32> to memref<*xi32>
    
    gpu.host_register %result0_cast : memref<*xi32>
    gpu.host_register %result1_cast : memref<*xi32>
    gpu.host_register %result2_cast : memref<*xi32>
    gpu.host_register %result3_cast : memref<*xi32>
    
    gpu.launch_func @kernels::@vector_gather blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) args(%result0 : memref<4xi32>, %result1 : memref<4xi32>, %result2 : memref<4xi32>, %result3 : memref<4xi32>)

    %result0_v = vector.load %result0[%c0] : memref<4xi32>, vector<4xi32>
    vector.print %result0_v : vector<4xi32>

    %result1_v = vector.load %result1[%c0] : memref<4xi32>, vector<4xi32>
    vector.print %result1_v : vector<4xi32>

    %result2_v = vector.load %result2[%c0] : memref<4xi32>, vector<4xi32>
    %result2_v_reshape = vector.shape_cast %result2_v : vector<4xi32> to vector<2x2xi32>
    vector.print %result2_v_reshape : vector<2x2xi32>

    %result3_v = vector.load %result3[%c0] : memref<4xi32>, vector<4xi32>
    %result3_v_reshape = vector.shape_cast %result3_v : vector<4xi32> to vector<2x2xi32>
    vector.print %result3_v_reshape : vector<2x2xi32>

    func.return
  }
}
