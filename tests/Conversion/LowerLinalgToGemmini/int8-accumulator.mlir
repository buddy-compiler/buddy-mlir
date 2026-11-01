// RUN: buddy-opt %s -convert-linalg-to-gemmini | FileCheck %s

// The matmul accumulates on the initial output: it is both D and C.
func.func @int8_accumulator(%a: memref<2x3xi8>, %b: memref<3x4xi8>, %out: memref<2x4xi32>) {
  linalg.matmul ins(%a, %b : memref<2x3xi8>, memref<3x4xi8>) outs(%out : memref<2x4xi32>)
  return
}

// CHECK-LABEL: func.func @int8_accumulator
// CHECK-SAME: %[[OUT:[a-z0-9]+]]: memref<2x4xi32>
// CHECK: gemmini.tile_matmul %{{.*}} %{{.*}} %[[OUT]] %[[OUT]]
// CHECK-SAME: fullC = true

// A [N, K] B (a Linear weight) is copied to [K, N] first.
#lhs = affine_map<(m, n, k) -> (m, k)>
#rhs = affine_map<(m, n, k) -> (n, k)>
#out = affine_map<(m, n, k) -> (m, n)>
func.func @transposed_b(%a: memref<2x3xi8>, %b: memref<4x3xi8>, %out: memref<2x4xi32>) {
  linalg.matmul indexing_maps = [#lhs, #rhs, #out] ins(%a, %b : memref<2x3xi8>, memref<4x3xi8>) outs(%out : memref<2x4xi32>)
  return
}
// CHECK-LABEL: func.func @transposed_b
// CHECK: %[[B:.*]] = memref.alloc() : memref<3x4xi8>
// CHECK: scf.for
// CHECK: gemmini.tile_matmul %{{.*}} %[[B]]
// CHECK-SAME: fullC = true

// Floating-point attention remains available to the CPU loop lowering.
func.func @float_attention(%a: memref<2x3xf32>, %b: memref<3x4xf32>, %out: memref<2x4xf32>) {
  linalg.matmul ins(%a, %b : memref<2x3xf32>, memref<3x4xf32>) outs(%out : memref<2x4xf32>)
  return
}
// CHECK-LABEL: func.func @float_attention
// CHECK-NOT: gemmini.tile_matmul
// CHECK: linalg.matmul
// CHECK: return

// The convolution loop writes scaled INT8 back; an INT32 output takes im2col
// and a matmul on the output, both D and C.
func.func @int8_convolution(%x: memref<1x9x9x5xi8>, %w: memref<7x3x3x5xi8>, %y: memref<1x4x4x7xi32>) {
  linalg.conv_2d_nhwc_fhwc {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%x, %w : memref<1x9x9x5xi8>, memref<7x3x3x5xi8>) outs(%y : memref<1x4x4x7xi32>)
  return
}
// CHECK-LABEL: func.func @int8_convolution
// CHECK: %[[COLUMNS:.*]] = memref.alloc() : memref<16x45xi8>
// CHECK: %[[Y:.*]] = memref.collapse_shape %{{.*}} {{\[\[}}0, 1, 2], [3]] : memref<1x4x4x7xi32> into memref<16x7xi32>
// CHECK: gemmini.tile_matmul %[[COLUMNS]] %{{.*}} %[[Y]] %[[Y]]
// CHECK-SAME: fullC = true
// CHECK-NOT: gemmini.tile_conv
