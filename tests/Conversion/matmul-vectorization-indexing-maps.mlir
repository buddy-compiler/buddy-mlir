// RUN: buddy-opt %s -matmul-vectorization | FileCheck %s

// CHECK-LABEL: func.func @transpose_b
// CHECK: linalg.matmul indexing_maps =
// CHECK-NOT: vector.load
// CHECK: return
func.func @transpose_b(%a: memref<1x17xf32>, %b: memref<11x17xf32>, %c: memref<1x11xf32>) {
  linalg.matmul indexing_maps = [affine_map<(m, n, k) -> (m, k)>, affine_map<(m, n, k) -> (n, k)>, affine_map<(m, n, k) -> (m, n)>]
      ins(%a, %b: memref<1x17xf32>, memref<11x17xf32>) outs(%c: memref<1x11xf32>)
  return
}

// CHECK-LABEL: func.func @standard
// CHECK-NOT: linalg.matmul
// CHECK: vector.load
// CHECK: return
func.func @standard(%a: memref<1x17xf32>, %b: memref<17x11xf32>, %c: memref<1x11xf32>) {
  linalg.matmul ins(%a, %b: memref<1x17xf32>, memref<17x11xf32>) outs(%c: memref<1x11xf32>)
  return
}
