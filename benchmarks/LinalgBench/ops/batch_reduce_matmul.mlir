// LinalgBench: linalg.batch_reduce_matmul (accumulate batched GEMMs into one output).
module {
  func.func @batch_reduce_matmul(%a: tensor<32x512x2048xf32>, %b: tensor<32x2048x512xf32>, %init: tensor<512x512xf32>) -> tensor<512x512xf32> {
    %0 = linalg.batch_reduce_matmul ins(%a, %b : tensor<32x512x2048xf32>, tensor<32x2048x512xf32>) outs(%init : tensor<512x512xf32>) -> tensor<512x512xf32>
    return %0 : tensor<512x512xf32>
  }
}
