// RUN: buddy-opt %s -lower-linalg-to-ime | FileCheck %s
//
// This file tests the lowering of linalg.matmul to ime.vmadot operations
// with boundary handling for non-aligned dimensions.
//

// =============================================================================
// Test case 1: Non-aligned M dimension (M=6, not divisible by TILE_M=4)
// =============================================================================
// CHECK-LABEL: func.func @matmul_i8_boundary_M
// CHECK: scf.for
// CHECK:   scf.for
// CHECK:     ime.vmadot
func.func @matmul_i8_boundary_M(%A: memref<6x8xi8>, %B: memref<8x4xi8>,
                                 %C: memref<6x4xi32>) {
  linalg.matmul ins(%A, %B : memref<6x8xi8>, memref<8x4xi8>)
                outs(%C : memref<6x4xi32>)
  return
}

// =============================================================================
// Test case 2: Non-aligned N dimension (N=6, not divisible by TILE_N=4)
// =============================================================================
// CHECK-LABEL: func.func @matmul_i8_boundary_N
// CHECK: scf.for
// CHECK:   scf.for
// CHECK:     ime.vmadot
func.func @matmul_i8_boundary_N(%A: memref<4x8xi8>, %B: memref<8x6xi8>,
                                 %C: memref<4x6xi32>) {
  linalg.matmul ins(%A, %B : memref<4x8xi8>, memref<8x6xi8>)
                outs(%C : memref<4x6xi32>)
  return
}

// =============================================================================
// Test case 3: Non-aligned K dimension (K=10, not divisible by TILE_K=8)
// =============================================================================
// CHECK-LABEL: func.func @matmul_i8_boundary_K
// CHECK: scf.for
// CHECK:   scf.for
// CHECK:     ime.vmadot
func.func @matmul_i8_boundary_K(%A: memref<4x10xi8>, %B: memref<10x4xi8>,
                                 %C: memref<4x4xi32>) {
  linalg.matmul ins(%A, %B : memref<4x10xi8>, memref<10x4xi8>)
                outs(%C : memref<4x4xi32>)
  return
}

// =============================================================================
// Test case 4: Multiple non-aligned dimensions (M=7, N=5, K=10)
// =============================================================================
// CHECK-LABEL: func.func @matmul_i8_boundary_all
// CHECK: scf.for
// CHECK:   scf.for
// CHECK:     ime.vmadot
func.func @matmul_i8_boundary_all(%A: memref<7x10xi8>, %B: memref<10x5xi8>,
                                   %C: memref<7x5xi32>) {
  linalg.matmul ins(%A, %B : memref<7x10xi8>, memref<10x5xi8>)
                outs(%C : memref<7x5xi32>)
  return
}

// =============================================================================
// Test case 5: Larger matrices with boundary (M=18, N=14, K=25)
// =============================================================================
// CHECK-LABEL: func.func @matmul_i8_large_boundary
// CHECK: scf.for
// CHECK:   scf.for
// CHECK:     ime.vmadot
func.func @matmul_i8_large_boundary(%A: memref<18x25xi8>, %B: memref<25x14xi8>,
                                     %C: memref<18x14xi32>) {
  linalg.matmul ins(%A, %B : memref<18x25xi8>, memref<25x14xi8>)
                outs(%C : memref<18x14xi32>)
  return
}

// =============================================================================
// Test case 6: Small matrices (all dimensions < tile size)
// =============================================================================
// M=3 < 4, N=2 < 4, K=5 < 8: Still uses vmadot with padding
// CHECK-LABEL: func.func @matmul_i8_small_all_scalar
// CHECK: ime.vmadot
func.func @matmul_i8_small_all_scalar(%A: memref<3x5xi8>, %B: memref<5x2xi8>,
                                       %C: memref<3x2xi32>) {
  linalg.matmul ins(%A, %B : memref<3x5xi8>, memref<5x2xi8>)
                outs(%C : memref<3x2xi32>)
  return
}

// =============================================================================
// Test case 7: int16 with boundary (tile size is 4x4x4 for int16)
// =============================================================================
// CHECK-LABEL: func.func @matmul_i16_boundary
// CHECK: scf.for
// CHECK:   scf.for
// CHECK:     ime.vmadot
func.func @matmul_i16_boundary(%A: memref<6x7xi16>, %B: memref<7x5xi16>,
                                %C: memref<6x5xi32>) {
  linalg.matmul ins(%A, %B : memref<6x7xi16>, memref<7x5xi16>)
                outs(%C : memref<6x5xi32>)
  return
}

// =============================================================================
// Test case 8: Edge case - exactly one tile with boundary
// =============================================================================
// CHECK-LABEL: func.func @matmul_i8_one_tile_plus
// CHECK: scf.for
// CHECK:   scf.for
// CHECK:     ime.vmadot
func.func @matmul_i8_one_tile_plus(%A: memref<5x8xi8>, %B: memref<8x4xi8>,
                                    %C: memref<5x4xi32>) {
  linalg.matmul ins(%A, %B : memref<5x8xi8>, memref<8x4xi8>)
                outs(%C : memref<5x4xi32>)
  return
}
