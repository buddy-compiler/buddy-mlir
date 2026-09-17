// RUN: buddy-opt %s -matmul-transpose-b-vectorization-decode="vector-size=4 unroll=1 n-tile=1" | FileCheck %s

// Square shapes alone cannot distinguish B[K,N] from B[N,K]. The decode
// pattern must check indexing maps before generating contiguous K loads.
// CHECK-LABEL: func.func @ordinary_square
// CHECK: linalg.matmul ins
// CHECK-NOT: vector.
// CHECK: return
func.func @ordinary_square(%a: memref<4x4xf32>, %b: memref<4x4xf32>,
                           %c: memref<4x4xf32>) {
  linalg.matmul ins(%a, %b : memref<4x4xf32>, memref<4x4xf32>)
                outs(%c : memref<4x4xf32>)
  return
}

// Unrelated rectangular matmuls must remain legal instead of failing partial
// conversion when they do not satisfy transpose-B shape constraints.
// CHECK-LABEL: func.func @ordinary_rectangular
// CHECK: linalg.matmul ins
// CHECK-NOT: vector.
// CHECK: return
func.func @ordinary_rectangular(%a: memref<2x5xf32>, %b: memref<5x3xf32>,
                                %c: memref<2x3xf32>) {
  linalg.matmul ins(%a, %b : memref<2x5xf32>, memref<5x3xf32>)
                outs(%c : memref<2x3xf32>)
  return
}

// The supported transpose-B form still lowers, including a K tail.
// CHECK-LABEL: func.func @transpose_b
// CHECK-NOT: linalg.matmul
// CHECK: vector.fma
// CHECK: vector.reduction <add>
// CHECK-NOT: linalg.matmul
// CHECK: return
func.func @transpose_b(%a: memref<2x5xf32>, %b: memref<3x5xf32>,
                       %c: memref<2x3xf32>) {
  linalg.matmul indexing_maps = [
    affine_map<(m, n, k) -> (m, k)>,
    affine_map<(m, n, k) -> (n, k)>,
    affine_map<(m, n, k) -> (m, n)>]
    ins(%a, %b : memref<2x5xf32>, memref<3x5xf32>)
    outs(%c : memref<2x3xf32>)
  return
}
