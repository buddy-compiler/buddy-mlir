// LinalgBench: linalg.pooling_ndhwc_max (volumetric, channels-last).
module {
  func.func @pooling_ndhwc_max(%input: tensor<1x12x40x40x32xf32>, %window: tensor<3x3x3xf32>, %init: tensor<1x10x38x38x32xf32>) -> tensor<1x10x38x38x32xf32> {
    %0 = linalg.pooling_ndhwc_max {dilations = dense<1> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>}
        ins(%input, %window : tensor<1x12x40x40x32xf32>, tensor<3x3x3xf32>) outs(%init : tensor<1x10x38x38x32xf32>) -> tensor<1x10x38x38x32xf32>
    return %0 : tensor<1x10x38x38x32xf32>
  }
}
