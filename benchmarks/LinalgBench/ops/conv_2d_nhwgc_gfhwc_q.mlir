// LinalgBench: linalg.conv_2d_nhwgc_gfhwc_q (INT8, NHWGC).
module {
  func.func @conv_2d_nhwgc_gfhwc_q(%input: tensor<1x56x56x32x4xi8>, %filter: tensor<32x8x3x3x4xi8>, %init: tensor<1x54x54x32x8xi32>) -> tensor<1x54x54x32x8xi32> {
    %zp = arith.constant 1 : i32
    %0 = linalg.conv_2d_nhwgc_gfhwc_q {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
        ins(%input, %filter, %zp, %zp : tensor<1x56x56x32x4xi8>, tensor<32x8x3x3x4xi8>, i32, i32)
        outs(%init : tensor<1x54x54x32x8xi32>) -> tensor<1x54x54x32x8xi32>
    return %0 : tensor<1x54x54x32x8xi32>
  }
}
