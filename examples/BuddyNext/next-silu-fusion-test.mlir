// RUN: buddy-opt %s \
// RUN:     -silu-fusion \
// RUN: | FileCheck %s

func.func private @rtclock() -> f64
func.func private @printMemrefF32(%ptr : tensor<*xf32>)

// Test SiLU fusion: sigmoid(x) * x -> fused SiLU
func.func @test_silu_fusion(%input: tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32> {
  %t_start = call @rtclock() : () -> f64

  // This pattern should be fused into a single linalg.generic
  %shift = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  %sigmoid = tosa.sigmoid %input : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
  %result = tosa.mul %input, %sigmoid, %shift : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>, tensor<1xi8>) -> tensor<1x40x11008xf32>

  %t_end = call @rtclock() : () -> f64
  %time = arith.subf %t_end, %t_start : f64

  %tensor_unranked = tensor.cast %result : tensor<1x40x11008xf32> to tensor<*xf32>
  call @printMemrefF32(%tensor_unranked) : (tensor<*xf32>) -> ()
  vector.print %time : f64

  return %result : tensor<1x40x11008xf32>
}

// Test reverse order: x * sigmoid(x) -> fused SiLU
func.func @test_silu_fusion_reverse(%input: tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32> {
  %shift = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  %sigmoid = tosa.sigmoid %input : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
  %result = tosa.mul %sigmoid, %input, %shift : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>, tensor<1xi8>) -> tensor<1x40x11008xf32>
  return %result : tensor<1x40x11008xf32>
}

// Test case that should NOT be fused (sigmoid has multiple uses)
func.func @test_no_fusion_multiple_uses(%input: tensor<1x40x11008xf32>) -> (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) {
  %shift = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  %sigmoid = tosa.sigmoid %input : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
  %result1 = tosa.mul %input, %sigmoid, %shift : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>, tensor<1xi8>) -> tensor<1x40x11008xf32>
  %result2 = tosa.add %sigmoid, %input : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
  return %result1, %result2 : tensor<1x40x11008xf32>, tensor<1x40x11008xf32>
}

func.func @main() {
  %input = arith.constant dense<2.0> : tensor<1x40x11008xf32>

  %result1 = call @test_silu_fusion(%input) : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
  %result2 = call @test_silu_fusion_reverse(%input) : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
  %result3, %result4 = call @test_no_fusion_multiple_uses(%input) : (tensor<1x40x11008xf32>) -> (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>)

  return
}

// CHECK-LABEL: func.func @test_silu_fusion
// CHECK-NOT: tosa.sigmoid
// CHECK-NOT: tosa.mul
// CHECK: %[[CST:.*]] = arith.constant 1.{{0+}}e+00 : f32
// CHECK: %[[EMPTY:.*]] = tensor.empty
// CHECK: %[[RESULT:.*]] = linalg.generic
// CHECK-SAME: ins(%arg0 : tensor<1x40x11008xf32>)
// CHECK-SAME: outs(%[[EMPTY]] : tensor<1x40x11008xf32>)
// CHECK: ^bb0(%[[IN:.*]]: f32, %[[OUT:.*]]: f32):
// CHECK: %[[NEG:.*]] = arith.negf %[[IN]] : f32
// CHECK: %[[EXP:.*]] = math.exp %[[NEG]] : f32
// CHECK: %[[ADD:.*]] = arith.addf %[[EXP]], %[[CST]] : f32
// CHECK: %[[DIV:.*]] = arith.divf %[[CST]], %[[ADD]] : f32
// CHECK: %[[MUL:.*]] = arith.mulf %[[IN]], %[[DIV]] : f32
// CHECK: linalg.yield %[[MUL]] : f32

// CHECK-LABEL: func.func @test_silu_fusion_reverse
// CHECK-NOT: tosa.sigmoid
// CHECK-NOT: tosa.mul
// CHECK: %[[RESULT:.*]] = linalg.generic
// CHECK: ^bb0(%[[IN:.*]]: f32, %[[OUT:.*]]: f32):
// CHECK: arith.negf
// CHECK: math.exp
// CHECK: arith.addf
// CHECK: arith.divf
// CHECK: arith.mulf
// CHECK: linalg.yield

// CHECK-LABEL: func.func @test_no_fusion_multiple_uses
// CHECK: tosa.sigmoid
// CHECK: tosa.mul
// CHECK: tosa.add
