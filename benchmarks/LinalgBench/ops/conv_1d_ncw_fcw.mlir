// LinalgBench: linalg.conv_1d_ncw_fcw (NCW × FCW), temporal / 1D-CNN style lengths.
module {
  func.func @conv_1d_ncw_fcw(%input: tensor<1x128x2048xf32>, %filter: tensor<256x128x7xf32>, %init: tensor<1x256x2042xf32>) -> tensor<1x256x2042xf32> {
    %0 = linalg.conv_1d_ncw_fcw {dilations = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>}
        ins(%input, %filter : tensor<1x128x2048xf32>, tensor<256x128x7xf32>)
        outs(%init : tensor<1x256x2042xf32>) -> tensor<1x256x2042xf32>
    return %0 : tensor<1x256x2042xf32>
  }
}
