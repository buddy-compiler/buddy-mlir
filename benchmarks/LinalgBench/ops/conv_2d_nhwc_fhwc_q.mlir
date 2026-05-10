// LinalgBench: linalg.conv_2d_nhwc_fhwc_q (INT8, FHWC weights).
module {
  func.func @conv_2d_nhwc_fhwc_q(%input: tensor<1x56x56x64xi8>, %filter: tensor<128x3x3x64xi8>, %init: tensor<1x54x54x128xi32>) -> tensor<1x54x54x128xi32> {
    %zp = arith.constant 1 : i32
    %0 = linalg.conv_2d_nhwc_fhwc_q {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
        ins(%input, %filter, %zp, %zp : tensor<1x56x56x64xi8>, tensor<128x3x3x64xi8>, i32, i32)
        outs(%init : tensor<1x54x54x128xi32>) -> tensor<1x54x54x128xi32>
    return %0 : tensor<1x54x54x128xi32>
  }
}
