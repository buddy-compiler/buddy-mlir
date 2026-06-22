// LinalgBench: linalg.conv_2d_nhwc_fhwc on tensors.
module {
  func.func @conv_2d_nhwc_fhwc(%input: tensor<1x64x64x32xf32>, %filter: tensor<64x3x3x32xf32>, %init: tensor<1x62x62x64xf32>) -> tensor<1x62x62x64xf32> {
    %0 = linalg.conv_2d_nhwc_fhwc
      ins(%input, %filter : tensor<1x64x64x32xf32>, tensor<64x3x3x32xf32>)
      outs(%init : tensor<1x62x62x64xf32>) -> tensor<1x62x62x64xf32>
    return %0 : tensor<1x62x62x64xf32>
  }
}
