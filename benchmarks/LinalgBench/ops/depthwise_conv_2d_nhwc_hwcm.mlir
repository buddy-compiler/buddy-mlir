// LinalgBench: linalg.depthwise_conv_2d_nhwc_hwcm (NHWC + channel multiplier, MobileNet-style).
module {
  func.func @depthwise_conv_2d_nhwc_hwcm(%input: tensor<1x56x56x128xf32>, %filter: tensor<3x3x128x2xf32>, %init: tensor<1x54x54x128x2xf32>) -> tensor<1x54x54x128x2xf32> {
    %0 = linalg.depthwise_conv_2d_nhwc_hwcm {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
        ins(%input, %filter : tensor<1x56x56x128xf32>, tensor<3x3x128x2xf32>) outs(%init : tensor<1x54x54x128x2xf32>) -> tensor<1x54x54x128x2xf32>
    return %0 : tensor<1x54x54x128x2xf32>
  }
}
