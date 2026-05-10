// LinalgBench: linalg.quantized_batch_matmul (batched INT8).
module {
  func.func @quantized_batch_matmul(%a: tensor<16x128x256xi8>, %b: tensor<16x256x128xi8>, %out: tensor<16x128x128xi32>) -> tensor<16x128x128xi32> {
    %zp = arith.constant 1 : i32
    %0 = linalg.quantized_batch_matmul ins(%a, %b, %zp, %zp : tensor<16x128x256xi8>, tensor<16x256x128xi8>, i32, i32)
        outs(%out : tensor<16x128x128xi32>) -> tensor<16x128x128xi32>
    return %0 : tensor<16x128x128xi32>
  }
}
