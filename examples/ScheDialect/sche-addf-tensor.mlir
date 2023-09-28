func.func @main() -> tensor<1x8x1x1xf32> attributes{sche.devices}{
  %0 = arith.constant dense<1.000000e-01> : tensor<1x8x1x1xf32>
  %1 = arith.constant dense<0.000000e+00> : tensor<1x8x1x1xf32>
  %2 = arith.addf %0, %1 : tensor<1x8x1x1xf32>
  return %2 : tensor<1x8x1x1xf32>
}