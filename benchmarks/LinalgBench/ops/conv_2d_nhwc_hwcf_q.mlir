// LinalgBench: linalg.conv_2d_nhwc_hwcf_q (INT8, channels-last activations).
module {
  func.func @conv_2d_nhwc_hwcf_q(%input: tensor<1x56x56x64xi8>, %filter: tensor<3x3x64x128xi8>, %init: tensor<1x54x54x128xi32>) -> tensor<1x54x54x128xi32> {
    %zp = arith.constant 1 : i32
    %0 = linalg.conv_2d_nhwc_hwcf_q {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
        ins(%input, %filter, %zp, %zp : tensor<1x56x56x64xi8>, tensor<3x3x64x128xi8>, i32, i32)
        outs(%init : tensor<1x54x54x128xi32>) -> tensor<1x54x54x128xi32>
    return %0 : tensor<1x54x54x128xi32>
  }
}
