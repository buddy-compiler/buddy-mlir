// LinalgBench: rank-specialized linalg.conv_3d (e.g. volumetric patch).
module {
  func.func @conv_3d_basic(%input: tensor<64x64x32xf32>, %filter: tensor<3x3x3xf32>, %init: tensor<62x62x30xf32>) -> tensor<62x62x30xf32> {
    %0 = linalg.conv_3d ins(%input, %filter : tensor<64x64x32xf32>, tensor<3x3x3xf32>) outs(%init : tensor<62x62x30xf32>) -> tensor<62x62x30xf32>
    return %0 : tensor<62x62x30xf32>
  }
}
