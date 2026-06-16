// LinalgBench: linalg.depthwise_conv_1d_ncw_cw (channels-first 1D map).
module {
  func.func @depthwise_conv_1d_ncw_cw(%input: tensor<1x128x4096xf32>, %filter: tensor<128x7xf32>, %init: tensor<1x128x4090xf32>) -> tensor<1x128x4090xf32> {
    %0 = linalg.depthwise_conv_1d_ncw_cw {dilations = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>}
        ins(%input, %filter : tensor<1x128x4096xf32>, tensor<128x7xf32>) outs(%init : tensor<1x128x4090xf32>) -> tensor<1x128x4090xf32>
    return %0 : tensor<1x128x4090xf32>
  }
}
