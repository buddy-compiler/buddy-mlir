// LinalgBench: linalg.softmax aggregate operation.
module {
  func.func @softmax(%input: tensor<8x16x32xf32>, %init: tensor<8x16x32xf32>) -> tensor<8x16x32xf32> {
    %0 = linalg.softmax dimension(2) ins(%input : tensor<8x16x32xf32>) outs(%init : tensor<8x16x32xf32>) -> tensor<8x16x32xf32>
    return %0 : tensor<8x16x32xf32>
  }
}
