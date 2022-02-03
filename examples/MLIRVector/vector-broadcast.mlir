memref.global "private" @gv : memref<4x4xf32> = dense<[[0. , 1. , 2. , 3. ],
                                                       [10., 11., 12., 13.],
                                                       [20., 21., 22., 23.],
                                                       [30., 31., 32., 33.]]>

func @main() {
  %mem = memref.get_global @gv : memref<4x4xf32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %f1 = arith.constant 1.0 : f32
  // Broadcast scalar to 1-D vector.
  %ele = memref.load %mem[%c1, %c1] : memref<4x4xf32>
  %broadcast_vec = vector.broadcast %ele : f32 to vector<4xf32>
  vector.print %broadcast_vec : vector<4xf32>
  // Broadcast 1-D vector to 2-D vector. 
  %load_vec = vector.load %mem[%c0, %c0] : memref<4x4xf32>, vector<4xf32>
  %broadcast_vec_2d = vector.broadcast %load_vec : vector<4xf32> to vector<4x4xf32>
  vector.print %broadcast_vec_2d : vector<4x4xf32>

  return
}
