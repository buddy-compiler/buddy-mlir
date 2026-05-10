// LinalgBench: linalg.vecmat on tensors.
module {
  func.func @vecmat(%x: tensor<512xf32>, %a: tensor<512x512xf32>, %init: tensor<512xf32>) -> tensor<512xf32> {
    %0 = linalg.vecmat ins(%x, %a : tensor<512xf32>, tensor<512x512xf32>) outs(%init : tensor<512xf32>) -> tensor<512xf32>
    return %0 : tensor<512xf32>
  }
}
