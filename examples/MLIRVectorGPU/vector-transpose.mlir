module attributes {gpu.container_module} {
  gpu.module @kernels {
    memref.global "private" @gv0 : memref<3x3xf32> = dense<[[0. , 1. , 2. ],
                                                            [10., 11., 12.],
                                                            [20., 21., 22.]]>
    memref.global "private" @gv1 : memref<3x4x5xf32> = dense<[[[0. , 1. , 2. , 3. , 4. ],
                                                              [10., 11., 12., 13., 14.],
                                                              [20., 21., 22., 23., 24.],
                                                              [30., 31., 32., 33., 34.]],
                                                             [[40., 41., 42., 43., 44.],
                                                              [50., 51., 52., 53., 54.],
                                                              [60., 61., 62., 63., 64.],
                                                              [70., 71., 72., 73., 74.]],
                                                             [[80., 81., 82., 83., 84.],
                                                              [90., 91., 92., 93., 94.],
                                                              [100., 101., 102., 103., 104.],
                                                              [110., 111., 112., 113., 114.]]]>
    gpu.func @vector_transpose(%result0: memref<9xf32>, %result1: memref<60xf32>) kernel {
      %c0 = arith.constant 0 : index
      %f0 = arith.constant 0.0 : f32
      %mem0 = memref.get_global @gv0 : memref<3x3xf32>
      %v0 = vector.transfer_read %mem0[%c0, %c0], %f0
        {permutation_map = affine_map<(d0, d1) -> (d0, d1)>} : memref<3x3xf32>, vector<3x3xf32>
      %v0_transposed = vector.transpose %v0, [1, 0] : vector<3x3xf32> to vector<3x3xf32>
      %v0_transposed_cast = vector.shape_cast %v0_transposed : vector<3x3xf32> to vector<9xf32>
      vector.store %v0_transposed_cast, %result0[%c0] : memref<9xf32>, vector<9xf32>

      %mem1 = memref.get_global @gv1 : memref<3x4x5xf32>
      %v1 = vector.transfer_read %mem1[%c0, %c0, %c0], %f0
        {permutation_map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>} : memref<3x4x5xf32>, vector<3x4x5xf32>
      %v1_transposed = vector.transpose %v1, [2, 0, 1] : vector<3x4x5xf32> to vector<5x3x4xf32>
      %v1_transposed_cast = vector.shape_cast %v1_transposed : vector<5x3x4xf32> to vector<60xf32>
      vector.store %v1_transposed_cast, %result1[%c0] : memref<60xf32>, vector<60xf32>

      gpu.return
    }
  }

  func.func @main() {
    %result0 = memref.alloc() : memref<9xf32>
    %result0_cast = memref.cast %result0 : memref<9xf32> to memref<*xf32>
    %result1 = memref.alloc() : memref<60xf32>
    %result1_cast = memref.cast %result1 : memref<60xf32> to memref<*xf32>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    gpu.host_register %result0_cast : memref<*xf32>
    gpu.host_register %result1_cast : memref<*xf32>
    gpu.launch_func @kernels::@vector_transpose blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) args(%result0 : memref<9xf32>, %result1 : memref<60xf32>)

    %result0_v = vector.load %result0[%c0] : memref<9xf32>, vector<9xf32>
    %result0_v_reshape = vector.shape_cast %result0_v : vector<9xf32> to vector<3x3xf32>

    %result1_v = vector.load %result1[%c0] : memref<60xf32>, vector<60xf32>
    %result1_v_reshape = vector.shape_cast %result1_v : vector<60xf32> to vector<5x3x4xf32>

    vector.print %result0_v_reshape : vector<3x3xf32>
    vector.print %result1_v_reshape : vector<5x3x4xf32>
    func.return
  }
}
