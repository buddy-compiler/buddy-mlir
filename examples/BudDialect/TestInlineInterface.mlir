memref.global "private" @gv : memref<4x4xf32> = dense<[[0. , 1. , 2. , 3. ],
                                                       [10., 11., 12., 13.],
                                                       [20., 21., 22., 23.],
                                                       [30., 31., 32., 33.]]>

func.func private @example(%arg0: memref<4x4xf32>) -> (f32) {
  %res = bud.test_array_attr %arg0 {coordinate = [0, 1]} : memref<4x4xf32>, f32
  return %res : f32
}

func.func @main() {
  %mem = memref.get_global @gv : memref<4x4xf32>
  %1 = func.call @example(%mem) : (memref<4x4xf32>) -> (f32)
  vector.print %1 : f32 
  return
}
