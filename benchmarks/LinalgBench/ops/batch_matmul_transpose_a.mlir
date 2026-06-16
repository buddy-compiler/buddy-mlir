// LinalgBench: linalg.batch_matmul_transpose_a (batched C = A^T B).
module {
  func.func @batch_matmul_transpose_a(%a_t: tensor<16x512x2048xf32>, %b: tensor<16x512x1024xf32>, %init: tensor<16x2048x1024xf32>) -> tensor<16x2048x1024xf32> {
    %0 = linalg.batch_matmul_transpose_a ins(%a_t, %b : tensor<16x512x2048xf32>, tensor<16x512x1024xf32>) outs(%init : tensor<16x2048x1024xf32>) -> tensor<16x2048x1024xf32>
    return %0 : tensor<16x2048x1024xf32>
  }
}
