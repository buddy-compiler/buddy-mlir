module {
  func.func @forward(%arg0: tensor<10x3x2xf32>, %arg1: tensor<10x3x2xf32>) -> tensor<10x3x2xf16> {
    %0 = tosa.cast %arg0 : (tensor<10x3x2xf32>) -> tensor<10x3x2xf16>
    %1 = tosa.cast %arg1 : (tensor<10x3x2xf32>) -> tensor<10x3x2xf16>
    %2 = tosa.add %0, %1 : (tensor<10x3x2xf16>, tensor<10x3x2xf16>) -> tensor<10x3x2xf16>
    return %2 : tensor<10x3x2xf16>
  }
}
// module {
//   func.func @forward(%arg0: tensor<10x3x2xf32>, %arg1: tensor<10x3x2xf32>) -> tensor<10x3x2xf32> {
//     %0 = tosa.cast %arg0 : (tensor<10x3x2xf32>) -> tensor<10x3x2xf32>
//     %1 = tosa.cast %arg1 : (tensor<10x3x2xf32>) -> tensor<10x3x2xf32>
//     %2 = tosa.add %0, %1 : (tensor<10x3x2xf32>, tensor<10x3x2xf32>) -> tensor<10x3x2xf32>
//     return %2 : tensor<10x3x2xf32>
//   }
// }