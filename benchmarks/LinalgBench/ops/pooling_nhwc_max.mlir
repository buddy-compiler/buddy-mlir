// LinalgBench: linalg.pooling_nhwc_max on tensors.
module {
  func.func @pooling_nhwc_max(%input: tensor<1x128x128x32xf32>, %window: tensor<2x2xf32>, %init: tensor<1x64x64x32xf32>) -> tensor<1x64x64x32xf32> {
    %0 = linalg.pooling_nhwc_max {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>}
      ins(%input, %window : tensor<1x128x128x32xf32>, tensor<2x2xf32>)
      outs(%init : tensor<1x64x64x32xf32>) -> tensor<1x64x64x32xf32>
    return %0 : tensor<1x64x64x32xf32>
  }
}
