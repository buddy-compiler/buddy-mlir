// LinalgBench: linalg.matvec on tensors.
module {
  func.func @matvec(%a: tensor<512x512xf32>, %x: tensor<512xf32>, %init: tensor<512xf32>) -> tensor<512xf32> {
    %0 = linalg.matvec ins(%a, %x : tensor<512x512xf32>, tensor<512xf32>) outs(%init : tensor<512xf32>) -> tensor<512xf32>
    return %0 : tensor<512xf32>
  }
}
