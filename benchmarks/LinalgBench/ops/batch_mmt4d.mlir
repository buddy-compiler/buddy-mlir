// LinalgBench: linalg.batch_mmt4d (blocked batched matmul / packing path).
module {
  func.func @batch_mmt4d(%a: tensor<32x8x16x4x1xf32>, %b: tensor<32x8x16x4x1xf32>, %init: tensor<32x8x8x4x4xf32>) -> tensor<32x8x8x4x4xf32> {
    %0 = linalg.batch_mmt4d ins(%a, %b : tensor<32x8x16x4x1xf32>, tensor<32x8x16x4x1xf32>) outs(%init : tensor<32x8x8x4x4xf32>) -> tensor<32x8x8x4x4xf32>
    return %0 : tensor<32x8x8x4x4xf32>
  }
}
