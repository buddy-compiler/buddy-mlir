// LinalgBench: rank-specialized linalg.conv_1d (e.g. long 1D sequence / audio frame).
module {
  func.func @conv_1d_basic(%input: tensor<8192xf32>, %filter: tensor<15xf32>, %init: tensor<8178xf32>) -> tensor<8178xf32> {
    %0 = linalg.conv_1d ins(%input, %filter : tensor<8192xf32>, tensor<15xf32>) outs(%init : tensor<8178xf32>) -> tensor<8178xf32>
    return %0 : tensor<8178xf32>
  }
}
