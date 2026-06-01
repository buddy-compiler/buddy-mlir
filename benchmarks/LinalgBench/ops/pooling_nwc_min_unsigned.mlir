// LinalgBench: linalg.pooling_nwc_min_unsigned (INT32, NWC).
module {
  func.func @pooling_nwc_min_unsigned(%input: tensor<1x2048x128xi32>, %window: tensor<3xi32>, %init: tensor<1x2046x128xi32>) -> tensor<1x2046x128xi32> {
    %0 = linalg.pooling_nwc_min_unsigned {dilations = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>}
        ins(%input, %window : tensor<1x2048x128xi32>, tensor<3xi32>) outs(%init : tensor<1x2046x128xi32>) -> tensor<1x2046x128xi32>
    return %0 : tensor<1x2046x128xi32>
  }
}
