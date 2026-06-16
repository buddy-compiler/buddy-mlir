// LinalgBench: linalg.mmt4d on tensors.
module {
  func.func @mmt4d(%a: tensor<16x16x8x4xf32>, %b: tensor<16x16x8x4xf32>, %init: tensor<16x16x8x8xf32>) -> tensor<16x16x8x8xf32> {
    %0 = linalg.mmt4d ins(%a, %b : tensor<16x16x8x4xf32>, tensor<16x16x8x4xf32>) outs(%init : tensor<16x16x8x8xf32>) -> tensor<16x16x8x8xf32>
    return %0 : tensor<16x16x8x8xf32>
  }
}
