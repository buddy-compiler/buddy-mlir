// LinalgBench: rank-specialized linalg.conv_2d (e.g. single-channel spatial map).
module {
  func.func @conv_2d_basic(%input: tensor<256x256xf32>, %filter: tensor<5x5xf32>, %init: tensor<252x252xf32>) -> tensor<252x252xf32> {
    %0 = linalg.conv_2d ins(%input, %filter : tensor<256x256xf32>, tensor<5x5xf32>) outs(%init : tensor<252x252xf32>) -> tensor<252x252xf32>
    return %0 : tensor<252x252xf32>
  }
}
