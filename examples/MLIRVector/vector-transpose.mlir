memref.global "private" @gv : memref<4x4xf32> = dense<[[0. , 1. , 2. , 3. ],
                                                       [10., 11., 12., 13.],
                                                       [20., 21., 22., 23.],
                                                       [30., 31., 32., 33.]]>

func.func @main() -> i32 {
  %mem = memref.get_global @gv : memref<4x4xf32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index

  // Load 2-D vector from memref.
  %f0 = arith.constant 0.0 : f32
  %vec_2d = vector.transfer_read %mem[%c0, %c0], %f0
      {permutation_map = affine_map<(d0, d1) -> (d0, d1)>} :
    memref<4x4xf32>, vector<2x4xf32>

  %vector_2_trans = vector.transpose %vec_2d, [1, 0] : vector<2x4xf32> to vector<4x2xf32> // Exchanges dims 0 and 1
  vector.print %vector_2_trans : vector<4x2xf32>

  %vector_2_same = vector.transpose %vec_2d, [0, 1] : vector<2x4xf32> to vector<2x4xf32> // Keeps the same vector dimensions
  vector.print %vector_2_same : vector<2x4xf32>

  %vector_3 = arith.constant dense<[[[45., 56., 78., 90., 12.], [23., 67., 90., 45., 54.], [23., 89., 100., 101., 114.], [123., 245., 67., 78., 90.]],
                                    [[451., 50., 79., 100., 12.], [29., 60., 91., 47., 50.], [28., 88., 109., 135., 104.], [123., 240., 64., 79., 99.]],
                                    [[45., 59., 77., 99., 121.], [25., 69., 99., 47., 58.], [13., 79., 101., 102., 115.], [124., 248., 671., 90., 234.]]]> : vector<3x4x5xf32>

  %vector_3_trans = vector.transpose %vector_3, [1, 2, 0] : vector<3x4x5xf32> to vector<4x5x3xf32>  // Changes all three dimensions
  vector.print %vector_3_trans : vector<4x5x3xf32>

  %ret = arith.constant 0 : i32
  return %ret : i32
}
