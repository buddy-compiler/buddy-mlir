// LinalgBench: linalg.transpose on tensors.
module {
  func.func @transpose(%input: tensor<64x128xf32>, %init: tensor<128x64xf32>) -> tensor<128x64xf32> {
    %0 = "linalg.transpose"(%input, %init) <{permutation = array<i64: 1, 0>}> ({
    ^bb0(%in: f32, %out: f32):
      "linalg.yield"(%in) : (f32) -> ()
    }) : (tensor<64x128xf32>, tensor<128x64xf32>) -> tensor<128x64xf32>
    return %0 : tensor<128x64xf32>
  }
}
