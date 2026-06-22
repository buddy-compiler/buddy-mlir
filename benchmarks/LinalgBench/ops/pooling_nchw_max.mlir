// LinalgBench: linalg.pooling_nchw_max (NCHW feature map).
module {
  func.func @pooling_nchw_max(%input: tensor<1x256x56x56xf32>, %window: tensor<3x3xf32>, %init: tensor<1x256x54x54xf32>) -> tensor<1x256x54x54xf32> {
    %0 = linalg.pooling_nchw_max {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
        ins(%input, %window : tensor<1x256x56x56xf32>, tensor<3x3xf32>) outs(%init : tensor<1x256x54x54xf32>) -> tensor<1x256x54x54xf32>
    return %0 : tensor<1x256x54x54xf32>
  }
}
