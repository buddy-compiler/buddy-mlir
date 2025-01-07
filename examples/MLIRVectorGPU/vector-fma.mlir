module attributes {gpu.container_module} {
  gpu.module @kernels {
    memref.global "private" @gv : memref<4x4xf32> = dense<[[0. , 1. , 2. , 3. ],
                                                          [10., 11., 12., 13.],
                                                          [20., 21., 22., 23.],
                                                          [30., 31., 32., 33.]]>
    gpu.func @vector_fma(%result: memref<4xf32>) kernel {
      %mem = memref.get_global @gv : memref<4x4xf32>
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %load_vec1 = vector.load %mem[%c0, %c0] : memref<4x4xf32>, vector<4xf32>
      %load_vec2 = vector.load %mem[%c1, %c0] : memref<4x4xf32>, vector<4xf32>
      %load_vec3 = vector.load %mem[%c2, %c0] : memref<4x4xf32>, vector<4xf32>
      %res = vector.fma %load_vec1, %load_vec2, %load_vec3 : vector<4xf32>
      vector.store %res, %result[%c0] : memref<4xf32>, vector<4xf32>
      gpu.return
    }
  }

  func.func @main() {
    %result = memref.alloc() : memref<4xf32>
    %result_cast = memref.cast %result : memref<4xf32> to memref<*xf32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    gpu.host_register %result_cast : memref<*xf32>
    gpu.launch_func @kernels::@vector_fma blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) args(%result : memref<4xf32>)
    %result_v = vector.load %result[%c0] : memref<4xf32>, vector<4xf32>
    vector.print %result_v : vector<4xf32>
    func.return
  }
}
