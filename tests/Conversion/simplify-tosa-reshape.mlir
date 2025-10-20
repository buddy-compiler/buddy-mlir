// RUN: buddy-opt -simplify-tosa-reshape %s | FileCheck %s

func.func @collapse_and_identity(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
  // Chain of reshapes: collapse + identity removal should result in returning %arg0.
  %r1 = tosa.reshape %arg0 {new_shape = array<i64: 6>} : (tensor<2x3xf32>) -> tensor<6xf32>
  %r2 = tosa.reshape %r1 {new_shape = array<i64: 2, 3>} : (tensor<6xf32>) -> tensor<2x3xf32>

  // Identity reshape (unused): should be erased.
  %id = tosa.reshape %arg0 {new_shape = array<i64: 2, 3>} : (tensor<2x3xf32>) -> tensor<2x3xf32>

  // Dead reshape (unused): should be erased.
  %dead = tosa.reshape %arg0 {new_shape = array<i64: 3, 2>} : (tensor<2x3xf32>) -> tensor<3x2xf32>

  return %r2 : tensor<2x3xf32>
}

// -----

func.func @keep_non_identity(%arg0: tensor<2x3xf32>) -> tensor<3x2xf32> {
  // Unused identity reshape: should be erased.
  %id = tosa.reshape %arg0 {new_shape = array<i64: 2, 3>} : (tensor<2x3xf32>) -> tensor<2x3xf32>

  // This non-identity reshape should be preserved.
  %r = tosa.reshape %arg0 {new_shape = array<i64: 3, 2>} : (tensor<2x3xf32>) -> tensor<3x2xf32>
  return %r : tensor<3x2xf32>
}

// CHECK-LABEL: func.func @collapse_and_identity
// CHECK-SAME:  (%arg0: tensor<2x3xf32>) -> tensor<2x3xf32>
// CHECK-NEXT:  return %arg0 : tensor<2x3xf32>
// CHECK-NEXT: }

// CHECK-LABEL: func.func @keep_non_identity
// CHECK-SAME:  (%arg0: tensor<2x3xf32>) -> tensor<3x2xf32>
// CHECK:       %[[R:.*]] = tosa.reshape %arg0 {new_shape = array<i64: 3, 2>} : (tensor<2x3xf32>) -> tensor<3x2xf32>
// CHECK-NEXT:  return %[[R]] : tensor<3x2xf32>
// CHECK-NEXT: }
