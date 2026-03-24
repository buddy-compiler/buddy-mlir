// RUN: buddy-opt %s -lower-linalg-to-ime | FileCheck %s
//
// This file tests the lowering of linalg.matmul (f16) to ime.vfmadot operations.
// FP16 tile sizes: TILE_M=4, TILE_K=4, TILE_N=4
// Output accumulator type is f16 (not i32).
//

// Test case 1: Minimal fp16 matmul, single tile (4x4) * (4x4) = (4x4)
// CHECK-LABEL: func.func @matmul_f16_4x4x4
// CHECK: scf.for
// CHECK:   scf.for
// CHECK:     scf.for
// CHECK:       ime.vfmadot
func.func @matmul_f16_4x4x4(%A: memref<4x4xf16>, %B: memref<4x4xf16>,
                              %C: memref<4x4xf16>) {
  linalg.matmul ins(%A, %B : memref<4x4xf16>, memref<4x4xf16>)
                outs(%C : memref<4x4xf16>)
  return
}

// Test case 2: Larger fp16 matmul requiring tiling (16x16) * (16x16) = (16x16)
// Each dimension needs 4 tiles along M, K, N.
// CHECK-LABEL: func.func @matmul_f16_16x16x16
// CHECK: scf.for
// CHECK:   scf.for
// CHECK:     scf.for
// CHECK:       ime.vfmadot
func.func @matmul_f16_16x16x16(%A: memref<16x16xf16>, %B: memref<16x16xf16>,
                                 %C: memref<16x16xf16>) {
  linalg.matmul ins(%A, %B : memref<16x16xf16>, memref<16x16xf16>)
                outs(%C : memref<16x16xf16>)
  return
}

// Test case 3: Non-square fp16 matmul (8x4) * (4x8) = (8x8)
// M=8 (2 tiles), K=4 (1 tile), N=8 (2 tiles)
// CHECK-LABEL: func.func @matmul_f16_8x4x8
// CHECK: scf.for
// CHECK:   scf.for
// CHECK:     scf.for
// CHECK:       ime.vfmadot
func.func @matmul_f16_8x4x8(%A: memref<8x4xf16>, %B: memref<4x8xf16>,
                              %C: memref<8x8xf16>) {
  linalg.matmul ins(%A, %B : memref<8x4xf16>, memref<4x8xf16>)
                outs(%C : memref<8x8xf16>)
  return
}

// Test case 4: Wide K dimension (4x16) * (16x4) = (4x4)
// K=16 means 4 tiles along reduction dimension.
// CHECK-LABEL: func.func @matmul_f16_4x16x4
// CHECK: scf.for
// CHECK:   scf.for
// CHECK:     scf.for
// CHECK:       ime.vfmadot
func.func @matmul_f16_4x16x4(%A: memref<4x16xf16>, %B: memref<16x4xf16>,
                               %C: memref<4x4xf16>) {
  linalg.matmul ins(%A, %B : memref<4x16xf16>, memref<16x4xf16>)
                outs(%C : memref<4x4xf16>)
  return
}
