// LinalgBench: linalg.depthwise_conv_1d_nwc_wc (channels-last 1D / per-channel temporal).
module {
  func.func @depthwise_conv_1d_nwc_wc(%input: tensor<1x4096x128xf32>, %filter: tensor<7x128xf32>, %init: tensor<1x4090x128xf32>) -> tensor<1x4090x128xf32> {
    %0 = linalg.depthwise_conv_1d_nwc_wc {dilations = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>}
        ins(%input, %filter : tensor<1x4096x128xf32>, tensor<7x128xf32>) outs(%init : tensor<1x4090x128xf32>) -> tensor<1x4090x128xf32>
    return %0 : tensor<1x4090x128xf32>
  }
}
