// LinalgBench: linalg.matmul on tensors.
module {
  func.func @matmul(%a: tensor<128x256xf32>, %b: tensor<256x128xf32>, %init: tensor<128x128xf32>) -> tensor<128x128xf32> {
    %0 = linalg.matmul ins(%a, %b : tensor<128x256xf32>, tensor<256x128xf32>) outs(%init : tensor<128x128xf32>) -> tensor<128x128xf32>
    return %0 : tensor<128x128xf32>
  }
}
