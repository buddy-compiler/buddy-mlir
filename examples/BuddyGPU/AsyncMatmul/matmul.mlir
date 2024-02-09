module {
  func.func @forward(%arg0: tensor<16x16xf32>, %arg1: tensor<16x16xf32>) -> tensor<16x16xf32> {
    %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
    %0 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%arg0, %arg1 : tensor<16x16xf32>, tensor<16x16xf32>) outs(%cst : tensor<16x16xf32>) -> tensor<16x16xf32>
    return %0 : tensor<16x16xf32>
  }
}

