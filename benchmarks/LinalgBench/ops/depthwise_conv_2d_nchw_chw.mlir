// LinalgBench: linalg.depthwise_conv_2d_nchw_chw (NCHW feature map).
module {
  func.func @depthwise_conv_2d_nchw_chw(%input: tensor<1x128x56x56xf32>, %filter: tensor<128x3x3xf32>, %init: tensor<1x128x54x54xf32>) -> tensor<1x128x54x54xf32> {
    %0 = linalg.depthwise_conv_2d_nchw_chw {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
        ins(%input, %filter : tensor<1x128x56x56xf32>, tensor<128x3x3xf32>) outs(%init : tensor<1x128x54x54xf32>) -> tensor<1x128x54x54xf32>
    return %0 : tensor<1x128x54x54xf32>
  }
}
