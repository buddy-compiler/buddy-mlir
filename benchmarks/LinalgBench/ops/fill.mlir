// LinalgBench: linalg.fill on tensors.
module {
  func.func @fill(%init: tensor<1024xf32>) -> tensor<1024xf32> {
    %cst = arith.constant 1.000000e+00 : f32
    %0 = linalg.fill ins(%cst : f32) outs(%init : tensor<1024xf32>) -> tensor<1024xf32>
    return %0 : tensor<1024xf32>
  }
}
