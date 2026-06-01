// LinalgBench: linalg.batch_vecmat (batched vector–matrix).
module {
  func.func @batch_vecmat(%x: tensor<32x512xf32>, %a: tensor<32x512x4096xf32>, %init: tensor<32x4096xf32>) -> tensor<32x4096xf32> {
    %0 = linalg.batch_vecmat ins(%x, %a : tensor<32x512xf32>, tensor<32x512x4096xf32>) outs(%init : tensor<32x4096xf32>) -> tensor<32x4096xf32>
    return %0 : tensor<32x4096xf32>
  }
}
