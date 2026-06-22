// LinalgBench: linalg.matmul_transpose_b on tensors.
module {
  func.func @matmul_transpose_b(%a: tensor<128x256xf32>, %b: tensor<128x256xf32>, %init: tensor<128x128xf32>) -> tensor<128x128xf32> {
    %0 = linalg.matmul_transpose_b ins(%a, %b : tensor<128x256xf32>, tensor<128x256xf32>) outs(%init : tensor<128x128xf32>) -> tensor<128x128xf32>
    return %0 : tensor<128x128xf32>
  }
}
