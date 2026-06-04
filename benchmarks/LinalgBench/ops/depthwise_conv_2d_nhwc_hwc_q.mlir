// LinalgBench: linalg.depthwise_conv_2d_nhwc_hwc_q (INT8 NHWC depthwise).
module {
  func.func @depthwise_conv_2d_nhwc_hwc_q(%input: tensor<1x56x56x128xi8>, %filter: tensor<3x3x128xi8>, %init: tensor<1x54x54x128xi32>) -> tensor<1x54x54x128xi32> {
    %zp = arith.constant 1 : i32
    %0 = linalg.depthwise_conv_2d_nhwc_hwc_q {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
        ins(%input, %filter, %zp, %zp : tensor<1x56x56x128xi8>, tensor<3x3x128xi8>, i32, i32)
        outs(%init : tensor<1x54x54x128xi32>) -> tensor<1x54x54x128xi32>
    return %0 : tensor<1x54x54x128xi32>
  }
}
