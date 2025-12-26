// RUN: buddy-opt -simplify-tosa-reshape %s | FileCheck %s

func.func @collapse_and_identity(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
  // Chain of reshapes: collapse + identity removal should result in returning %arg0.
  %s0 = tosa.const_shape {values = dense<[6]> : tensor<1xindex>} : () -> !tosa.shape<1>
  %r1 = tosa.reshape %arg0, %s0 : (tensor<2x3xf32>, !tosa.shape<1>) -> tensor<6xf32>
  %s1 = tosa.const_shape {values = dense<[2, 3]> : tensor<2xindex>} : () -> !tosa.shape<2>
  %r2 = tosa.reshape %r1, %s1 : (tensor<6xf32>, !tosa.shape<2>) -> tensor<2x3xf32>

  // Identity reshape (unused): should be erased.
  %s2 = tosa.const_shape {values = dense<[2, 3]> : tensor<2xindex>} : () -> !tosa.shape<2>
  %id = tosa.reshape %arg0, %s2 : (tensor<2x3xf32>, !tosa.shape<2>) -> tensor<2x3xf32>

  // Dead reshape (unused): should be erased.
  %s3 = tosa.const_shape {values = dense<[3, 2]> : tensor<2xindex>} : () -> !tosa.shape<2>
  %dead = tosa.reshape %arg0, %s3 : (tensor<2x3xf32>, !tosa.shape<2>) -> tensor<3x2xf32>

  return %r2 : tensor<2x3xf32>
}

// -----

func.func @keep_non_identity(%arg0: tensor<2x3xf32>) -> tensor<3x2xf32> {
  // Unused identity reshape: should be erased.
  %s0 = tosa.const_shape {values = dense<[2, 3]> : tensor<2xindex>} : () -> !tosa.shape<2>
  %id = tosa.reshape %arg0, %s0 : (tensor<2x3xf32>, !tosa.shape<2>) -> tensor<2x3xf32>

  // This non-identity reshape should be preserved.
  %s1 = tosa.const_shape {values = dense<[3, 2]> : tensor<2xindex>} : () -> !tosa.shape<2>
  %r = tosa.reshape %arg0, %s1 : (tensor<2x3xf32>, !tosa.shape<2>) -> tensor<3x2xf32>
  return %r : tensor<3x2xf32>
}

// CHECK-LABEL: func.func @collapse_and_identity
// CHECK-SAME:  (%arg0: tensor<2x3xf32>) -> tensor<2x3xf32>
// CHECK:       tosa.const_shape
// CHECK:       tosa.const_shape
// CHECK:       tosa.const_shape
// CHECK:       tosa.const_shape
// CHECK:       return %arg0 : tensor<2x3xf32>
// CHECK-NEXT: }

// CHECK-LABEL: func.func @keep_non_identity
// CHECK-SAME:  (%arg0: tensor<2x3xf32>) -> tensor<3x2xf32>
// CHECK:       tosa.const_shape
// CHECK:       %[[S:.*]] = tosa.const_shape {values = dense<[3, 2]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK-NEXT:  %[[R:.*]] = tosa.reshape %arg0, %[[S]] : (tensor<2x3xf32>, !tosa.shape<2>) -> tensor<3x2xf32>
// CHECK-NEXT:  return %[[R]] : tensor<3x2xf32>
// CHECK-NEXT: }
