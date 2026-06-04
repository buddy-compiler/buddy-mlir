// LinalgBench: grouped linalg.conv_2d_ngchw_fgchw (MobileNet-style depth grouping).
module {
  func.func @conv_2d_ngchw_fgchw(%input: tensor<1x32x4x56x56xf32>, %filter: tensor<8x32x4x3x3xf32>, %init: tensor<1x32x8x54x54xf32>) -> tensor<1x32x8x54x54xf32> {
    %0 = linalg.conv_2d_ngchw_fgchw {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
        ins(%input, %filter : tensor<1x32x4x56x56xf32>, tensor<8x32x4x3x3xf32>)
        outs(%init : tensor<1x32x8x54x54xf32>) -> tensor<1x32x8x54x54xf32>
    return %0 : tensor<1x32x8x54x54xf32>
  }
}
