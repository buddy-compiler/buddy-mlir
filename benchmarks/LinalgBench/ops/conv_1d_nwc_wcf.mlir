// LinalgBench: linalg.conv_1d_nwc_wcf on tensors.
module {
  func.func @conv_1d_nwc_wcf(%input: tensor<8x1024x16xf32>, %filter: tensor<3x16x32xf32>, %init: tensor<8x1022x32xf32>) -> tensor<8x1022x32xf32> {
    %0 = linalg.conv_1d_nwc_wcf
      ins(%input, %filter : tensor<8x1024x16xf32>, tensor<3x16x32xf32>)
      outs(%init : tensor<8x1022x32xf32>) -> tensor<8x1022x32xf32>
    return %0 : tensor<8x1022x32xf32>
  }
}
