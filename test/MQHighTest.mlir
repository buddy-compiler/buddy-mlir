//===- MQHighTest.mlir - Test MQHigh dialect operations -----------===//
//
// This file tests the operations for the MQHigh dialect.
//
//===----------------------------------------------------------------------===//

module {
  // Test MatMul operation
  func.func @test_matmul(%A: tensor<2x3xf32>, %B: tensor<3x4xf32>) -> tensor<2x4xf32> {
    %result = mqhigh.matmul %A, %B : tensor<2x3xf32>, tensor<3x4xf32>
    return %result : tensor<2x4xf32>
  }

  // Test Pack operation
  func.func @test_pack(%input: tensor<2x3xf32>) -> tensor<2x3xf32> {
    %result = mqhigh.pack %input : tensor<2x3xf32>
    return %result : tensor<2x3xf32>
  }

  // Test Unpack operation
  func.func @test_unpack(%input: tensor<2x3xf32>) -> tensor<2x3xf32> {
    %result = mqhigh.unpack %input : tensor<2x3xf32>
    return %result : tensor<2x3xf32>
  }
}
