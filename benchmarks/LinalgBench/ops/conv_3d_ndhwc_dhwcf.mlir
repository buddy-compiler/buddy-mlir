// LinalgBench: linalg.conv_3d_ndhwc_dhwcf (dense 3D + channels-last, short clip / patch).
module {
  func.func @conv_3d_ndhwc_dhwcf(%input: tensor<1x16x64x64x32xf32>, %filter: tensor<3x3x3x32x64xf32>, %init: tensor<1x14x62x62x64xf32>) -> tensor<1x14x62x62x64xf32> {
    %0 = linalg.conv_3d_ndhwc_dhwcf {dilations = dense<1> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>}
        ins(%input, %filter : tensor<1x16x64x64x32xf32>, tensor<3x3x3x32x64xf32>)
        outs(%init : tensor<1x14x62x62x64xf32>) -> tensor<1x14x62x62x64xf32>
    return %0 : tensor<1x14x62x62x64xf32>
  }
}
