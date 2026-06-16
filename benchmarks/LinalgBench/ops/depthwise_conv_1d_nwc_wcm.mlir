// LinalgBench: linalg.depthwise_conv_1d_nwc_wcm (1D depthwise + channel multiplier).
module {
  func.func @depthwise_conv_1d_nwc_wcm(%input: tensor<1x4096x128xf32>, %filter: tensor<7x128x2xf32>, %init: tensor<1x4090x128x2xf32>) -> tensor<1x4090x128x2xf32> {
    %0 = linalg.depthwise_conv_1d_nwc_wcm {dilations = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>}
        ins(%input, %filter : tensor<1x4096x128xf32>, tensor<7x128x2xf32>) outs(%init : tensor<1x4090x128x2xf32>) -> tensor<1x4090x128x2xf32>
    return %0 : tensor<1x4090x128x2xf32>
  }
}
