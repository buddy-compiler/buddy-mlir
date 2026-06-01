// LinalgBench: linalg.dot on tensors.
module {
  func.func @dot(%a: tensor<4096xf32>, %b: tensor<4096xf32>, %init: tensor<f32>) -> tensor<f32> {
    %0 = linalg.dot ins(%a, %b : tensor<4096xf32>, tensor<4096xf32>) outs(%init : tensor<f32>) -> tensor<f32>
    return %0 : tensor<f32>
  }
}
