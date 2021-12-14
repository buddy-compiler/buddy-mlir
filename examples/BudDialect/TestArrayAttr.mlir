module {
  memref.global "private" @gv : memref<4x4xf32> = dense<[[0. , 1. , 2. , 3. ],
                                                         [10., 11., 12., 13.],
                                                         [20., 21., 22., 23.],
                                                         [30., 31., 32., 33.]]>
  %mem = memref.get_global @gv : memref<4x4xf32>
  %res = bud.test_array_attr %mem {coordinate = [0, 1]} : memref<4x4xf32>, f32
}
