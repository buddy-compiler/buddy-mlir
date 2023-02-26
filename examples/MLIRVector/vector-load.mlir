memref.global "private" @gv : memref<4x4xf32> = dense<[[0. , 1. , 2. , 3. ],
                                                       [10., 11., 12., 13.],
                                                       [20., 21., 22., 23.],
                                                       [30., 31., 32., 33.]]>

func.func @main() -> i32 {
  %mem = memref.get_global @gv : memref<4x4xf32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  // Load 1-D vector from memref.
  %vec_1d = vector.load %mem[%c0, %c0] : memref<4x4xf32>, vector<4xf32>
  vector.print %vec_1d : vector<4xf32>
  // Load 2-D vector from memref.
  %f0 = arith.constant 0.0 : f32
  %vec_2d = vector.transfer_read %mem[%c2, %c0], %f0
      {permutation_map = affine_map<(d0, d1) -> (d0, d1)>} :
    memref<4x4xf32>, vector<2x4xf32>
  vector.print %vec_2d : vector<2x4xf32>

  %ret = arith.constant 0 : i32
  return %ret : i32
}
