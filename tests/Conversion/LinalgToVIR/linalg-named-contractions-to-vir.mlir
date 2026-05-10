// RUN: buddy-opt %s -split-input-file -lower-linalg-to-vir | FileCheck %s

func.func @matmul_transpose_a(%a: memref<4x8xf32>, %b: memref<4x16xf32>, %c: memref<8x16xf32>) {
  linalg.matmul_transpose_a
      ins(%a, %b : memref<4x8xf32>, memref<4x16xf32>)
      outs(%c : memref<8x16xf32>)
  return
}

// CHECK-LABEL: func.func @matmul_transpose_a
// CHECK-NOT: linalg.matmul_transpose_a
// CHECK: vir.set_vl
// CHECK: vir.load
// CHECK: vir.broadcast
// CHECK: vir.fma
// CHECK: vir.store

// -----

func.func @batch_matvec(%a: memref<2x4x8xf32>, %x: memref<2x8xf32>, %y: memref<2x4xf32>) {
  linalg.batch_matvec
      ins(%a, %x : memref<2x4x8xf32>, memref<2x8xf32>)
      outs(%y : memref<2x4xf32>)
  return
}

// CHECK-LABEL: func.func @batch_matvec
// CHECK-NOT: linalg.batch_matvec
// CHECK: vir.set_vl
// CHECK: memref.transpose
// CHECK: vir.fma
// CHECK: vir.store

// -----

func.func @batch_vecmat(%x: memref<2x4xf32>, %b: memref<2x4x8xf32>, %y: memref<2x8xf32>) {
  linalg.batch_vecmat
      ins(%x, %b : memref<2x4xf32>, memref<2x4x8xf32>)
      outs(%y : memref<2x8xf32>)
  return
}

// CHECK-LABEL: func.func @batch_vecmat
// CHECK-NOT: linalg.batch_vecmat
// CHECK: vir.set_vl
// CHECK: vir.broadcast
// CHECK: vir.fma
// CHECK: vir.store

// -----

func.func @batch_reduce_matmul(%a: memref<2x4x8xf32>, %b: memref<2x8x4xf32>, %c: memref<4x4xf32>) {
  linalg.batch_reduce_matmul
      ins(%a, %b : memref<2x4x8xf32>, memref<2x8x4xf32>)
      outs(%c : memref<4x4xf32>)
  return
}

// CHECK-LABEL: func.func @batch_reduce_matmul
// CHECK-NOT: linalg.batch_reduce_matmul
// CHECK: vir.set_vl
// CHECK: affine.for
// CHECK: vir.fma
// CHECK: vir.store

// -----

func.func @mmt4d(%a: memref<2x4x2x2xf32>, %b: memref<3x4x4x2xf32>, %c: memref<2x3x2x4xf32>) {
  linalg.mmt4d
      ins(%a, %b : memref<2x4x2x2xf32>, memref<3x4x4x2xf32>)
      outs(%c : memref<2x3x2x4xf32>)
  return
}

// CHECK-LABEL: func.func @mmt4d
// CHECK-NOT: linalg.mmt4d
// CHECK: vir.set_vl
// CHECK: memref.transpose
// CHECK: vir.fma
// CHECK: vir.store

// -----

func.func @batch_mmt4d(%a: memref<2x2x4x2x2xf32>, %b: memref<2x3x4x4x2xf32>, %c: memref<2x2x3x2x4xf32>) {
  linalg.batch_mmt4d
      ins(%a, %b : memref<2x2x4x2x2xf32>, memref<2x3x4x4x2xf32>)
      outs(%c : memref<2x2x3x2x4xf32>)
  return
}

// CHECK-LABEL: func.func @batch_mmt4d
// CHECK-NOT: linalg.batch_mmt4d
// CHECK: vir.set_vl
// CHECK: memref.transpose
// CHECK: vir.fma
// CHECK: vir.store
