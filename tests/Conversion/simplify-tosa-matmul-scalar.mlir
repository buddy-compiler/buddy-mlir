// RUN: buddy-opt -simplify-tosa-matmul-scalar %s | FileCheck %s

// Case: K=1, J=1, batch dims equal → matmul should become mul.
func.func @scalar_like_matmul(%arg0: tensor<1x64x1xf32>, %arg1: tensor<1x1x1xf32>) -> tensor<1x64x1xf32> {
  %a_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
  %b_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
  %0 = tosa.matmul %arg0, %arg1, %a_zp, %b_zp : (tensor<1x64x1xf32>, tensor<1x1x1xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x64x1xf32>
  return %0 : tensor<1x64x1xf32>
}

// -----

// Case: Not rewritten when J != 1.
func.func @non_scalar_j(%arg0: tensor<1x64x1xf32>, %arg1: tensor<1x1x2xf32>) -> tensor<1x64x2xf32> {
  %a_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
  %b_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
  %0 = tosa.matmul %arg0, %arg1, %a_zp, %b_zp : (tensor<1x64x1xf32>, tensor<1x1x2xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x64x2xf32>
  return %0 : tensor<1x64x2xf32>
}

// -----

// Case: Not rewritten when K != 1.
func.func @non_scalar_k(%arg0: tensor<1x64x3xf32>, %arg1: tensor<1x3x1xf32>) -> tensor<1x64x1xf32> {
  %a_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
  %b_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
  %0 = tosa.matmul %arg0, %arg1, %a_zp, %b_zp : (tensor<1x64x3xf32>, tensor<1x3x1xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x64x1xf32>
  return %0 : tensor<1x64x1xf32>
}

// CHECK-LABEL: func.func @scalar_like_matmul
// CHECK-SAME:  (%arg0: tensor<1x64x1xf32>, %arg1: tensor<1x1x1xf32>) -> tensor<1x64x1xf32>
// CHECK:       %[[SHIFT:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK:       %[[MUL:.*]] = tosa.mul %arg0, %arg1, %[[SHIFT]] : (tensor<1x64x1xf32>, tensor<1x1x1xf32>, tensor<1xi8>) -> tensor<1x64x1xf32>
// CHECK-NEXT:  return %[[MUL]] : tensor<1x64x1xf32>
// CHECK-NOT:   tosa.matmul
// CHECK-NEXT: }

// CHECK-LABEL: func.func @non_scalar_j
// CHECK:       tosa.matmul

// CHECK-LABEL: func.func @non_scalar_k
// CHECK:       tosa.matmul
