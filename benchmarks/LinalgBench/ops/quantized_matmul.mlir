// LinalgBench: linalg.quantized_matmul (INT8 GEMM-style).
module {
  func.func @quantized_matmul(%a: tensor<512x2048xi8>, %b: tensor<2048x512xi8>, %out: tensor<512x512xi32>) -> tensor<512x512xi32> {
    %zp = arith.constant 1 : i32
    %0 = linalg.quantized_matmul ins(%a, %b, %zp, %zp : tensor<512x2048xi8>, tensor<2048x512xi8>, i32, i32)
        outs(%out : tensor<512x512xi32>) -> tensor<512x512xi32>
    return %0 : tensor<512x512xi32>
  }
}
