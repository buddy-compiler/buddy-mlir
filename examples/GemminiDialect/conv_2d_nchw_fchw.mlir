memref.global "private" @input : memref<1x2x5x5xi8> = dense<[[[[1, 0, -1, 0, 1],
                                                               [1, 0, -1, 0, 1],
                                                               [1, 0, -1, 0, 1],
                                                               [1, 0, -1, 0, 1],
                                                               [-1, 0, 1, 0, -1]],
                                                              [[-1, 0, 1, 0, -1],
                                                               [-1, 0, 1, 0, -1],
                                                               [-1, 0, 1, 0, -1],
                                                               [-1, 0, 1, 0, -1],
                                                               [-1, 0, 1, 0, -1]]]]>

memref.global "private" @weight : memref<1x2x3x3xi8> = dense<[[[[1, 2, 3],
                                                                [3, 2, 1],
                                                                [1, 2, 3]],
                                                               [[3, 2, 1],
                                                                [1, 2, 3],
                                                                [3, 2, 1]]]]>

func.func @main() -> i8 {
  %0 = arith.constant 0 : i8
  %mem0 = memref.get_global @input  : memref<1x2x5x5xi8> 
  %mem1 = memref.get_global @weight : memref<1x2x3x3xi8>
  %mem2 = memref.alloc() : memref<1x1x3x3xi8> 
  linalg.conv_2d_nchw_fchw 
    ins (%mem0, %mem1 : memref<1x2x5x5xi8>, memref<1x2x3x3xi8>)
  outs(%mem2 : memref<1x1x3x3xi8>)
  gemmini.print %mem2 : memref<1x1x3x3xi8>
  return %0 : i8
}
