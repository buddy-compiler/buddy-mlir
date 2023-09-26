func.func @conv_2d_nchw_fchw(%arg0: memref<?x?x?x?xf32>, %arg1: memref<?x?x?x?xf32>, %arg2: memref<?x?x?x?xf32>) {
    linalg.conv_2d_nchw_fchw ins (%arg0, %arg1: memref<?x?x?x?xf32>, memref<?x?x?x?xf32>)
                             outs (%arg2: memref<?x?x?x?xf32>)
    return
}