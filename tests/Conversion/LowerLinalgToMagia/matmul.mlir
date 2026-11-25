// RUN: buddy-opt %s -convert-linalg-to-magia | FileCheck %s

// A product read as [K, N]: the operands go to FP16, the mesh computes, the
// result comes back to FP32.
// CHECK-LABEL: func.func @product
// CHECK: %[[X:.*]] = memref.alloc() : memref<32x768xf16>
// CHECK: arith.truncf
// CHECK: %[[W:.*]] = memref.alloc() : memref<768x64xf16>
// CHECK: %[[Z:.*]] = memref.alloc() : memref<32x64xf16>
// CHECK: magia.gemm %[[X]] %[[W]] %[[Z]] : memref<32x768xf16> memref<768x64xf16> memref<32x64xf16>
// CHECK: arith.extf
// CHECK: memref.dealloc %[[X]]
// CHECK-NOT: linalg.matmul
func.func @product(%a: memref<32x768xf32>, %b: memref<768x64xf32>, %c: memref<32x64xf32>) {
  linalg.matmul ins(%a, %b : memref<32x768xf32>, memref<768x64xf32>) outs(%c : memref<32x64xf32>)
  return
}

// A Linear keeps its weight as [N, K]: W is its transpose.
// CHECK-LABEL: func.func @linear
// CHECK: %[[W:.*]] = memref.alloc() : memref<768x64xf16>
// CHECK: magia.gemm %{{.*}} %[[W]] %{{.*}} : memref<32x768xf16> memref<768x64xf16> memref<32x64xf16>
func.func @linear(%a: memref<32x768xf32>, %b: memref<64x768xf32>, %c: memref<32x64xf32>) {
  linalg.matmul indexing_maps = [affine_map<(m, n, k) -> (m, k)>, affine_map<(m, n, k) -> (n, k)>, affine_map<(m, n, k) -> (m, n)>]
    ins(%a, %b : memref<32x768xf32>, memref<64x768xf32>) outs(%c : memref<32x64xf32>)
  return
}

// A weight kept in FP16 as [N, K] is W in place, wherever it lies in its pack.
// CHECK-LABEL: func.func @weight
// CHECK: %[[X:.*]] = memref.alloc() : memref<32x768xf16>
// CHECK: %[[Z:.*]] = memref.alloc() : memref<32x64xf16>
// CHECK: magia.gemm %[[X]] %[[B:.*]] %[[Z]] : memref<32x768xf16> memref<768x64xf16, strided<[?, ?], offset: ?>> memref<32x64xf16>
// CHECK-NOT: memref.dealloc %[[B]]
// CHECK-NOT: linalg.matmul
func.func @weight(%a: memref<32x768xf32>, %b: memref<768x64xf16, strided<[?, ?], offset: ?>>, %c: memref<32x64xf32>) {
  linalg.matmul ins(%a, %b : memref<32x768xf32>, memref<768x64xf16, strided<[?, ?], offset: ?>>) outs(%c : memref<32x64xf32>)
  return
}

// The products of a batch, one after another.
// CHECK-LABEL: func.func @batch
// CHECK: magia.gemm
// CHECK: magia.gemm
// CHECK-NOT: linalg.batch_matmul
func.func @batch(%a: memref<2x32x64xf32>, %b: memref<2x64x32xf32>, %c: memref<2x32x32xf32>) {
  linalg.batch_matmul ins(%a, %b : memref<2x32x64xf32>, memref<2x64x32xf32>) outs(%c : memref<2x32x32xf32>)
  return
}

// The rows of a batch's matrices may be dynamic.
// CHECK-LABEL: func.func @rows
// CHECK: magia.gemm %{{.*}} %{{.*}} %{{.*}} : memref<?x64xf16> memref<64x32xf16> memref<?x32xf16>
// CHECK: magia.gemm
// CHECK-NOT: linalg.batch_matmul
func.func @rows(%a: memref<2x?x64xf32>, %b: memref<2x64x32xf32>, %c: memref<2x?x32xf32>) {
  linalg.batch_matmul ins(%a, %b : memref<2x?x64xf32>, memref<2x64x32xf32>) outs(%c : memref<2x?x32xf32>)
  return
}

// INT8 products are not the mesh's.
// CHECK-LABEL: func.func @integers
// CHECK: linalg.matmul
func.func @integers(%a: memref<4x8xi8>, %b: memref<8x4xi8>, %c: memref<4x4xi32>) {
  linalg.matmul ins(%a, %b : memref<4x8xi8>, memref<8x4xi8>) outs(%c : memref<4x4xi32>)
  return
}
