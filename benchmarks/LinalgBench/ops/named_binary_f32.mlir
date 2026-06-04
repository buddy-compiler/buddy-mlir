// LinalgBench: binary f32 named elementwise operations.
module {
  func.func @named_binary_f32(%a: tensor<1024xf32>, %b: tensor<1024xf32>,
                              %init: tensor<1024xf32>) -> tensor<1024xf32> {
    %0 = linalg.add ins(%a, %b : tensor<1024xf32>, tensor<1024xf32>) outs(%init : tensor<1024xf32>) -> tensor<1024xf32>
    %1 = linalg.sub ins(%0, %b : tensor<1024xf32>, tensor<1024xf32>) outs(%init : tensor<1024xf32>) -> tensor<1024xf32>
    %2 = linalg.mul ins(%1, %b : tensor<1024xf32>, tensor<1024xf32>) outs(%init : tensor<1024xf32>) -> tensor<1024xf32>
    %3 = linalg.div ins(%2, %b : tensor<1024xf32>, tensor<1024xf32>) outs(%init : tensor<1024xf32>) -> tensor<1024xf32>
    %4 = linalg.max ins(%3, %a : tensor<1024xf32>, tensor<1024xf32>) outs(%init : tensor<1024xf32>) -> tensor<1024xf32>
    %5 = linalg.min ins(%4, %b : tensor<1024xf32>, tensor<1024xf32>) outs(%init : tensor<1024xf32>) -> tensor<1024xf32>
    %6 = linalg.powf ins(%5, %b : tensor<1024xf32>, tensor<1024xf32>) outs(%init : tensor<1024xf32>) -> tensor<1024xf32>
    return %6 : tensor<1024xf32>
  }
}
