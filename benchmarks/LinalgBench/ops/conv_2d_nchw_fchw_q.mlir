// LinalgBench: linalg.conv_2d_nchw_fchw_q (INT8, NCHW).
module {
  func.func @conv_2d_nchw_fchw_q(%input: tensor<1x64x56x56xi8>, %filter: tensor<128x64x3x3xi8>, %init: tensor<1x128x54x54xi32>) -> tensor<1x128x54x54xi32> {
    %zp = arith.constant 1 : i32
    %0 = linalg.conv_2d_nchw_fchw_q {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
        ins(%input, %filter, %zp, %zp : tensor<1x64x56x56xi8>, tensor<128x64x3x3xi8>, i32, i32)
        outs(%init : tensor<1x128x54x54xi32>) -> tensor<1x128x54x54xi32>
    return %0 : tensor<1x128x54x54xi32>
  }
}
