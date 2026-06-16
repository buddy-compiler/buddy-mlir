// LinalgBench: linalg.depthwise_conv_3d_ndhwc_dhwcm (NDHWC depthwise + channel multiplier).
module {
  func.func @depthwise_conv_3d_ndhwc_dhwcm(%input: tensor<1x16x32x32x64xf32>, %filter: tensor<3x3x3x64x2xf32>, %init: tensor<1x14x30x30x64x2xf32>) -> tensor<1x14x30x30x64x2xf32> {
    %0 = linalg.depthwise_conv_3d_ndhwc_dhwcm {dilations = dense<1> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>}
        ins(%input, %filter : tensor<1x16x32x32x64xf32>, tensor<3x3x3x64x2xf32>) outs(%init : tensor<1x14x30x30x64x2xf32>) -> tensor<1x14x30x30x64x2xf32>
    return %0 : tensor<1x14x30x30x64x2xf32>
  }
}
