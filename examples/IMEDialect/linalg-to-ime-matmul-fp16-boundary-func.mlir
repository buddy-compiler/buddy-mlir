// RUN: buddy-opt %s -lower-linalg-to-ime | FileCheck %s
//
// Test case: C[7x5] = A[7x6] * B[6x5] with fp16, non-aligned dimensions
// FP16 tile sizes: TILE_M=4, TILE_K=4, TILE_N=4
//   M=7: 1 full tile (4) + 3 remaining
//   K=6: 1 full tile (4) + 2 remaining
//   N=5: 1 full tile (4) + 1 remaining
//
// This file is designed to also be compiled and linked with
// runtime_matmul_fp16_boundary.c for functional verification on hardware.

// CHECK-LABEL: func.func @matmul_fp16_boundary
// CHECK: scf.for
// CHECK:   scf.for
// CHECK:     ime.vfmadot
func.func @matmul_fp16_boundary(%A: memref<7x6xf16>, %B: memref<6x5xf16>,
                                 %C: memref<7x5xf16>) {
  linalg.matmul ins(%A, %B : memref<7x6xf16>, memref<6x5xf16>)
                outs(%C : memref<7x5xf16>)
  return
}
