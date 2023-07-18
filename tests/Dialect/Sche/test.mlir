module {

  func.func @main() -> tensor<1x8x1x1xf32> {
    %0 = arith.constant dense<1.000000e-01> : tensor<1x8x1x1xf32>
    %1 = arith.constant dense<0.000000e+00> : tensor<1x8x1x1xf32>
    // %0,%1 = sche.ondevice () {targetId:"cpu", targetConfig=""} ()->(tensor<1x8x1x1xf32>, tensor<1x8x1x1xf32>) {
    //     %0 = arith.constant dense<1.000000e-01> : tensor<1x8x1x1xf32>
    //     %1 = arith.constant dense<0.000000e+00> : tensor<1x8x1x1xf32>
    //     return %0,%1
    // }
    %2 = arith.addf %0, %1 : tensor<1x8x1x1xf32>
    // %2 = sche.ondevice (%0, %1) {targetId:"gpu", targetConfig=""} (%args0, %args1 : tensor<1x8x1x1xf32>, tensor<1x8x1x1xf32>)->(tensor<1x8x1x1xf32>) {
    //     %0 = arith.addf %args0, %args1 : tensor<1x8x1x1xf32>
    //     return %0
    // }
    return %2 : tensor<1x8x1x1xf32>
  }
}