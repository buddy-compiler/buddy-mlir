// LinalgBench: unary f32 named elementwise operations.
module {
  func.func @named_unary_f32(%arg: tensor<1024xf32>, %init: tensor<1024xf32>) -> tensor<1024xf32> {
    %0 = linalg.exp ins(%arg : tensor<1024xf32>) outs(%init : tensor<1024xf32>) -> tensor<1024xf32>
    %1 = linalg.log ins(%0 : tensor<1024xf32>) outs(%init : tensor<1024xf32>) -> tensor<1024xf32>
    %2 = linalg.abs ins(%1 : tensor<1024xf32>) outs(%init : tensor<1024xf32>) -> tensor<1024xf32>
    %3 = linalg.ceil ins(%2 : tensor<1024xf32>) outs(%init : tensor<1024xf32>) -> tensor<1024xf32>
    %4 = linalg.floor ins(%3 : tensor<1024xf32>) outs(%init : tensor<1024xf32>) -> tensor<1024xf32>
    %5 = linalg.negf ins(%4 : tensor<1024xf32>) outs(%init : tensor<1024xf32>) -> tensor<1024xf32>
    %6 = linalg.reciprocal ins(%5 : tensor<1024xf32>) outs(%init : tensor<1024xf32>) -> tensor<1024xf32>
    %7 = linalg.round ins(%6 : tensor<1024xf32>) outs(%init : tensor<1024xf32>) -> tensor<1024xf32>
    %8 = linalg.sqrt ins(%7 : tensor<1024xf32>) outs(%init : tensor<1024xf32>) -> tensor<1024xf32>
    %9 = linalg.rsqrt ins(%8 : tensor<1024xf32>) outs(%init : tensor<1024xf32>) -> tensor<1024xf32>
    %10 = linalg.square ins(%9 : tensor<1024xf32>) outs(%init : tensor<1024xf32>) -> tensor<1024xf32>
    %11 = linalg.tanh ins(%10 : tensor<1024xf32>) outs(%init : tensor<1024xf32>) -> tensor<1024xf32>
    %12 = linalg.erf ins(%11 : tensor<1024xf32>) outs(%init : tensor<1024xf32>) -> tensor<1024xf32>
    return %12 : tensor<1024xf32>
  }
}
