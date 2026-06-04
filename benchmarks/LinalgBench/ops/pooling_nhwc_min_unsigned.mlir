// LinalgBench: linalg.pooling_nhwc_min_unsigned (INT32 activations).
module {
  func.func @pooling_nhwc_min_unsigned(%input: tensor<1x56x56x128xi32>, %window: tensor<3x3xi32>, %init: tensor<1x54x54x128xi32>) -> tensor<1x54x54x128xi32> {
    %0 = linalg.pooling_nhwc_min_unsigned {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
        ins(%input, %window : tensor<1x56x56x128xi32>, tensor<3x3xi32>) outs(%init : tensor<1x54x54x128xi32>) -> tensor<1x54x54x128xi32>
    return %0 : tensor<1x54x54x128xi32>
  }
}
