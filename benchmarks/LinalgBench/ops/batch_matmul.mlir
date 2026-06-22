// LinalgBench: linalg.batch_matmul on tensors.
module {
  func.func @batch_matmul(%a: tensor<16x64x128xf32>, %b: tensor<16x128x64xf32>, %init: tensor<16x64x64xf32>) -> tensor<16x64x64xf32> {
    %0 = linalg.batch_matmul ins(%a, %b : tensor<16x64x128xf32>, tensor<16x128x64xf32>) outs(%init : tensor<16x64x64xf32>) -> tensor<16x64x64xf32>
    return %0 : tensor<16x64x64xf32>
  }
}
