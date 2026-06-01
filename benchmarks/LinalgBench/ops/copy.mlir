// LinalgBench: linalg.copy on tensors.
module {
  func.func @copy(%input: tensor<1024xf32>, %init: tensor<1024xf32>) -> tensor<1024xf32> {
    %0 = linalg.copy ins(%input : tensor<1024xf32>) outs(%init : tensor<1024xf32>) -> tensor<1024xf32>
    return %0 : tensor<1024xf32>
  }
}
