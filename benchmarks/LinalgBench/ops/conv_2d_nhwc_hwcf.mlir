// LinalgBench: linalg.conv_2d_nhwc_hwcf on tensors.
module {
  func.func @conv_2d_nhwc_hwcf(%input: tensor<1x64x64x32xf32>, %filter: tensor<3x3x32x64xf32>, %init: tensor<1x62x62x64xf32>) -> tensor<1x62x62x64xf32> {
    %0 = linalg.conv_2d_nhwc_hwcf
      ins(%input, %filter : tensor<1x64x64x32xf32>, tensor<3x3x32x64xf32>)
      outs(%init : tensor<1x62x62x64xf32>) -> tensor<1x62x62x64xf32>
    return %0 : tensor<1x62x62x64xf32>
  }
}
