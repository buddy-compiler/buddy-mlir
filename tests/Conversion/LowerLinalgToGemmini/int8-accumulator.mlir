// RUN: buddy-opt %s -convert-linalg-to-gemmini | FileCheck %s

func.func @int8_accumulator(%a: memref<2x3xi8>, %b: memref<3x4xi8>, %out: memref<2x4xi32>) {
  linalg.matmul ins(%a, %b : memref<2x3xi8>, memref<3x4xi8>) outs(%out : memref<2x4xi32>)
  return
}

// CHECK-LABEL: func.func @int8_accumulator
// CHECK: memref.alloc() : memref<2x4xi32>
// CHECK: gemmini.tile_matmul
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
