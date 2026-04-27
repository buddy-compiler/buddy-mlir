// RUN: buddy-opt %s -lower-linalg-to-ime | FileCheck %s
//
// This file tests fp16 linalg.matmul -> ime.vfmadot lowering
// for matrices with dimensions NOT aligned to TILE_M=4, TILE_K=4, TILE_N=4.
// The boundary-handling pattern pads tiles with zeros and masks writes.
//

// =============================================================================
// Test case 1: Non-aligned M dimension (M=6, not divisible by TILE_M=4)
// =============================================================================
// CHECK-LABEL: func.func @matmul_f16_boundary_M
// CHECK: scf.for
// CHECK:   scf.for
// CHECK:     ime.vfmadot
func.func @matmul_f16_boundary_M(%A: memref<6x4xf16>, %B: memref<4x4xf16>,
                                   %C: memref<6x4xf16>) {
  linalg.matmul ins(%A, %B : memref<6x4xf16>, memref<4x4xf16>)
                outs(%C : memref<6x4xf16>)
  return
}

// =============================================================================
// Test case 2: Non-aligned N dimension (N=6, not divisible by TILE_N=4)
// =============================================================================
// CHECK-LABEL: func.func @matmul_f16_boundary_N
// CHECK: scf.for
// CHECK:   scf.for
// CHECK:     ime.vfmadot
func.func @matmul_f16_boundary_N(%A: memref<4x4xf16>, %B: memref<4x6xf16>,
                                   %C: memref<4x6xf16>) {
  linalg.matmul ins(%A, %B : memref<4x4xf16>, memref<4x6xf16>)
                outs(%C : memref<4x6xf16>)
  return
}

// =============================================================================
// Test case 3: Non-aligned K dimension (K=6, not divisible by TILE_K=4)
// =============================================================================
// CHECK-LABEL: func.func @matmul_f16_boundary_K
// CHECK: scf.for
// CHECK:   scf.for
// CHECK:     ime.vfmadot
func.func @matmul_f16_boundary_K(%A: memref<4x6xf16>, %B: memref<6x4xf16>,
                                   %C: memref<4x4xf16>) {
  linalg.matmul ins(%A, %B : memref<4x6xf16>, memref<6x4xf16>)
                outs(%C : memref<4x4xf16>)
  return
}

// =============================================================================
// Test case 4: All dimensions non-aligned (M=7, K=6, N=5)
// M=7: 1 full tile (4) + 3 remaining
// K=6: 1 full tile (4) + 2 remaining
// N=5: 1 full tile (4) + 1 remaining
// =============================================================================
// CHECK-LABEL: func.func @matmul_f16_boundary_all
// CHECK: scf.for
// CHECK:   scf.for
// CHECK:     ime.vfmadot
func.func @matmul_f16_boundary_all(%A: memref<7x6xf16>, %B: memref<6x5xf16>,
                                    %C: memref<7x5xf16>) {
  linalg.matmul ins(%A, %B : memref<7x6xf16>, memref<6x5xf16>)
                outs(%C : memref<7x5xf16>)
  return
}
