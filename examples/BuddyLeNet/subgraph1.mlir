module {
  func.func @subgraph1(%arg0: tensor<120x256xf32>, %arg1: tensor<84x120xf32>, %arg2: tensor<10x84xf32>) -> (tensor<256x120xf32>, tensor<120x84xf32>, tensor<84x10xf32>) {
    %0 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1 = tosa.transpose %arg0, %0 : (tensor<120x256xf32>, tensor<2xi32>) -> tensor<256x120xf32>
    %2 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3 = tosa.transpose %arg1, %2 : (tensor<84x120xf32>, tensor<2xi32>) -> tensor<120x84xf32>
    %4 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %5 = tosa.transpose %arg2, %4 : (tensor<10x84xf32>, tensor<2xi32>) -> tensor<84x10xf32>
    return %1, %3, %5 : tensor<256x120xf32>, tensor<120x84xf32>, tensor<84x10xf32>
  }
}

