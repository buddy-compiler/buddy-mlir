// LinalgBench: grouped linalg.conv_2d_nhwgc_gfhwc (NHWGC activations).
module {
  func.func @conv_2d_nhwgc_gfhwc(%input: tensor<1x56x56x32x4xf32>, %filter: tensor<32x8x3x3x4xf32>, %init: tensor<1x54x54x32x8xf32>) -> tensor<1x54x54x32x8xf32> {
    %0 = linalg.conv_2d_nhwgc_gfhwc {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
        ins(%input, %filter : tensor<1x56x56x32x4xf32>, tensor<32x8x3x3x4xf32>)
        outs(%init : tensor<1x54x54x32x8xf32>) -> tensor<1x54x54x32x8xf32>
    return %0 : tensor<1x54x54x32x8xf32>
  }
}
