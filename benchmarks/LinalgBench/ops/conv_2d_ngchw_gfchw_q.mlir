// LinalgBench: linalg.conv_2d_ngchw_gfchw_q (INT8 grouped, GFCHW).
module {
  func.func @conv_2d_ngchw_gfchw_q(%input: tensor<1x32x4x56x56xi8>, %filter: tensor<32x8x4x3x3xi8>, %init: tensor<1x32x8x54x54xi32>) -> tensor<1x32x8x54x54xi32> {
    %zp = arith.constant 1 : i32
    %0 = linalg.conv_2d_ngchw_gfchw_q {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
        ins(%input, %filter, %zp, %zp : tensor<1x32x4x56x56xi8>, tensor<32x8x4x3x3xi8>, i32, i32)
        outs(%init : tensor<1x32x8x54x54xi32>) -> tensor<1x32x8x54x54xi32>
    return %0 : tensor<1x32x8x54x54xi32>
  }
}
