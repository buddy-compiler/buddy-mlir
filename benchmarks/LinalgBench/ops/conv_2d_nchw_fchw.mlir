// LinalgBench: linalg.conv_2d_nchw_fchw on tensors.
module {
  func.func @conv_2d_nchw_fchw(%input: tensor<1x32x64x64xf32>, %filter: tensor<64x32x3x3xf32>, %init: tensor<1x64x62x62xf32>) -> tensor<1x64x62x62xf32> {
    %0 = linalg.conv_2d_nchw_fchw
      ins(%input, %filter : tensor<1x32x64x64xf32>, tensor<64x32x3x3xf32>)
      outs(%init : tensor<1x64x62x62xf32>) -> tensor<1x64x62x62xf32>
    return %0 : tensor<1x64x62x62xf32>
  }
}
