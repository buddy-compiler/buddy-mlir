// LinalgBench: linalg.pooling_ncw_max (NCW layout).
module {
  func.func @pooling_ncw_max(%input: tensor<1x128x2048xf32>, %window: tensor<3xf32>, %init: tensor<1x128x2046xf32>) -> tensor<1x128x2046xf32> {
    %0 = linalg.pooling_ncw_max {dilations = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>}
        ins(%input, %window : tensor<1x128x2048xf32>, tensor<3xf32>) outs(%init : tensor<1x128x2046xf32>) -> tensor<1x128x2046xf32>
    return %0 : tensor<1x128x2046xf32>
  }
}
