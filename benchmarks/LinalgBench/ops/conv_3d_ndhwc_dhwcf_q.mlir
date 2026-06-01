// LinalgBench: linalg.conv_3d_ndhwc_dhwcf_q (INT8 NDHWC, volumetric patch).
module {
  func.func @conv_3d_ndhwc_dhwcf_q(%input: tensor<1x16x64x64x32xi8>, %filter: tensor<3x3x3x32x64xi8>, %init: tensor<1x14x62x62x64xi32>) -> tensor<1x14x62x62x64xi32> {
    %zp = arith.constant 1 : i32
    %0 = linalg.conv_3d_ndhwc_dhwcf_q {dilations = dense<1> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>}
        ins(%input, %filter, %zp, %zp : tensor<1x16x64x64x32xi8>, tensor<3x3x3x32x64xi8>, i32, i32)
        outs(%init : tensor<1x14x62x62x64xi32>) -> tensor<1x14x62x62x64xi32>
    return %0 : tensor<1x14x62x62x64xi32>
  }
}
