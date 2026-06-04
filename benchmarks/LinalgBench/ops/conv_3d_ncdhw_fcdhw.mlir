// LinalgBench: linalg.conv_3d_ncdhw_fcdhw (channels-first 3D, e.g. video / volumetric block).
module {
  func.func @conv_3d_ncdhw_fcdhw(%input: tensor<1x64x8x32x32xf32>, %filter: tensor<128x64x3x3x3xf32>, %init: tensor<1x128x6x30x30xf32>) -> tensor<1x128x6x30x30xf32> {
    %0 = linalg.conv_3d_ncdhw_fcdhw {dilations = dense<1> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>}
        ins(%input, %filter : tensor<1x64x8x32x32xf32>, tensor<128x64x3x3x3xf32>)
        outs(%init : tensor<1x128x6x30x30xf32>) -> tensor<1x128x6x30x30xf32>
    return %0 : tensor<1x128x6x30x30xf32>
  }
}
