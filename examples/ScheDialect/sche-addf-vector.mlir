func.func @main() -> vector<1x8x1x1xf32> {
  %0 = arith.constant dense<1.000000e-01> : vector<1x8x1x1xf32>
  %1 = arith.constant dense<0.000000e+00> : vector<1x8x1x1xf32>
  %2 = arith.addf %0, %1 : vector<1x8x1x1xf32>
  return %2 : vector<1x8x1x1xf32>
}