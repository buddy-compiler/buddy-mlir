// LinalgBench: linalg.depthwise_conv_2d_nhwc_hwc on tensors.
module {
  func.func @depthwise_conv_2d_nhwc_hwc(%input: tensor<1x64x64x32xf32>, %filter: tensor<3x3x32xf32>, %init: tensor<1x62x62x32xf32>) -> tensor<1x62x62x32xf32> {
    %0 = linalg.depthwise_conv_2d_nhwc_hwc
      ins(%input, %filter : tensor<1x64x64x32xf32>, tensor<3x3x32xf32>)
      outs(%init : tensor<1x62x62x32xf32>) -> tensor<1x62x62x32xf32>
    return %0 : tensor<1x62x62x32xf32>
  }
}
