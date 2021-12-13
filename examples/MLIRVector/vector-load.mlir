memref.global "private" @gv : memref<4x4xf32> = dense<[[0. , 1. , 2. , 3. ],
                                                       [10., 11., 12., 13.],
                                                       [20., 21., 22., 23.],
                                                       [30., 31., 32., 33.]]>

func @main() {
  %mem = memref.get_global @gv : memref<4x4xf32>
  %c0 = arith.constant 0 : index
  %0 = vector.load %mem[%c0, %c0] : memref<4x4xf32>, vector<4xf32>
  vector.print %0 : vector<4xf32>
  return
}
