// LinalgBench: linalg.depthwise_conv_3d_ncdhw_cdhw (volumetric, channels-first).
module {
  func.func @depthwise_conv_3d_ncdhw_cdhw(%input: tensor<1x64x16x32x32xf32>, %filter: tensor<64x3x3x3xf32>, %init: tensor<1x64x14x30x30xf32>) -> tensor<1x64x14x30x30xf32> {
    %0 = linalg.depthwise_conv_3d_ncdhw_cdhw {dilations = dense<1> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>}
        ins(%input, %filter : tensor<1x64x16x32x32xf32>, tensor<64x3x3x3xf32>) outs(%init : tensor<1x64x14x30x30xf32>) -> tensor<1x64x14x30x30xf32>
    return %0 : tensor<1x64x14x30x30xf32>
  }
}
