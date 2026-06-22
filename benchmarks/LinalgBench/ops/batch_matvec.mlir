// LinalgBench: linalg.batch_matvec (batched matrix–vector).
module {
  func.func @batch_matvec(%a: tensor<32x512x4096xf32>, %x: tensor<32x4096xf32>, %init: tensor<32x512xf32>) -> tensor<32x512xf32> {
    %0 = linalg.batch_matvec ins(%a, %x : tensor<32x512x4096xf32>, tensor<32x4096xf32>) outs(%init : tensor<32x512xf32>) -> tensor<32x512xf32>
    return %0 : tensor<32x512xf32>
  }
}
