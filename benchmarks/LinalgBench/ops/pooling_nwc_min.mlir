// LinalgBench: linalg.pooling_nwc_min (NWC layout).
module {
  func.func @pooling_nwc_min(%input: tensor<1x2048x128xf32>, %window: tensor<3xf32>, %init: tensor<1x2046x128xf32>) -> tensor<1x2046x128xf32> {
    %0 = linalg.pooling_nwc_min {dilations = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>}
        ins(%input, %window : tensor<1x2048x128xf32>, tensor<3xf32>) outs(%init : tensor<1x2046x128xf32>) -> tensor<1x2046x128xf32>
    return %0 : tensor<1x2046x128xf32>
  }
}
