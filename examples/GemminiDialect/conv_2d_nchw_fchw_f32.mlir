memref.global "private" @input : memref<2x2x5x5xf32> = dense<[[[[1., 0., -1., 0., 1.],
                                                               [1., 0., -1., 0., 1.],
                                                               [1., 0., -1., 0., 1.],
                                                               [1., 0., -1., 0., 1.],
                                                               [-1., 0., 1., 0., -1.]],
                                                              [[-1., 0., 1., 0., -1.],
                                                               [-1., 0., 1., 0., -1.],
                                                               [-1., 0., 1., 0., -1.],
                                                               [-1., 0., 1., 0., -1.],
                                                               [-1., 0., 1., 0., -1.]]],
                                                             [[[1., 0., 2., 0., 1.],
                                                               [1., 0., 2., 0., 1.],
                                                               [1., 0., 2., 0., 1.],
                                                               [1., 0., 2., 0., 1.],
                                                               [-1., 0., 2., 0., -1.]],
                                                              [[-1., 0., 2., 0., -1.],
                                                               [-1., 0., 2., 0., -1.],
                                                               [-1., 0., 2., 0., -1.],
                                                               [-1., 0., 2., 0., -1.],
                                                               [-1., 0., 2., 0., -1.]]]]>

memref.global "private" @weight : memref<2x2x3x3xf32> = dense<[[[[1., 2., 3.],
                                                                [3., 2., 1.],
                                                                [1., 2., 3.]],
                                                               [[3., 2., 1.],
                                                                [1., 2., 3.],
                                                                [3., 2., 1.]]],
                                                                [[[1., 2., 3.],
                                                                [3., 2., 1.],
                                                                [1., 2., 3.]],
                                                               [[3., 2., 1.],
                                                                [1., 2., 3.],
                                                                [3., 2., 1.]]]]>

func.func @main() -> i8 {
  %0 = arith.constant 0 : i8
  %mem0 = memref.get_global @input  : memref<2x2x5x5xf32> 
  %mem1 = memref.get_global @weight : memref<2x2x3x3xf32>
  %mem2 = memref.alloc() : memref<2x2x3x3xf32> 
  linalg.conv_2d_nchw_fchw 
    ins (%mem0, %mem1 : memref<2x2x5x5xf32>, memref<2x2x3x3xf32>)
  outs(%mem2 : memref<2x2x3x3xf32>)
  gemmini.print %mem2 : memref<2x2x3x3xf32>
  return %0 : i8
}
