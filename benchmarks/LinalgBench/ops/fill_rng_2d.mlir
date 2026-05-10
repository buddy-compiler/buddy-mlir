// LinalgBench: linalg.fill_rng_2d.
module {
  func.func @fill_rng_2d(%init: tensor<64x64xf32>) -> tensor<64x64xf32> {
    %min = arith.constant 0.000000e+00 : f64
    %max = arith.constant 1.000000e+00 : f64
    %seed = arith.constant 7 : i32
    %0 = linalg.fill_rng_2d ins(%min, %max, %seed : f64, f64, i32) outs(%init : tensor<64x64xf32>) -> tensor<64x64xf32>
    return %0 : tensor<64x64xf32>
  }
}
