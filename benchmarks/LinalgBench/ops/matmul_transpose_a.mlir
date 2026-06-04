// LinalgBench: linalg.matmul_transpose_a (C = A^T B), GEMM-shaped tiles.
module {
  func.func @matmul_transpose_a(%a_t: tensor<512x2048xf32>, %b: tensor<512x1024xf32>, %init: tensor<2048x1024xf32>) -> tensor<2048x1024xf32> {
    %0 = linalg.matmul_transpose_a ins(%a_t, %b : tensor<512x2048xf32>, tensor<512x1024xf32>) outs(%init : tensor<2048x1024xf32>) -> tensor<2048x1024xf32>
    return %0 : tensor<2048x1024xf32>
  }
}
