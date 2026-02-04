// RUN: buddy-opt %s -lower-linalg-to-ime | FileCheck %s
// 
// This file tests the lowering of linalg.matmul to ime.vmadot operations.
//

// Test case 1: Simple int8 matmul (4x8) * (8x4) = (4x4)
// This maps to ime.vmadot with single iteration loops
// CHECK-LABEL: func.func @matmul_i8_4x8x4
// CHECK: scf.for
// CHECK:   scf.for
// CHECK:     scf.for
// CHECK:       memref.subview {{.*}} [4, 8]
// CHECK:       memref.subview {{.*}} [8, 4]
// CHECK:       memref.subview {{.*}} [4, 4]
// CHECK:       ime.vmadot
func.func @matmul_i8_4x8x4(%A: memref<4x8xi8>, %B: memref<8x4xi8>, %C: memref<4x4xi32>) {
  linalg.matmul ins(%A, %B : memref<4x8xi8>, memref<8x4xi8>)
                outs(%C : memref<4x4xi32>)
  return
}

// Test case 2: Larger int8 matmul requiring tiling
// (16x32) * (32x16) = (16x16)
// Should generate nested loops with ime.vmadot
// CHECK-LABEL: func.func @matmul_i8_16x32x16
// CHECK: scf.for
// CHECK:   scf.for
// CHECK:     scf.for
// CHECK:       memref.subview {{.*}} [4, 8]
// CHECK:       memref.subview {{.*}} [8, 4]
// CHECK:       memref.subview {{.*}} [4, 4]
// CHECK:       ime.vmadot
func.func @matmul_i8_16x32x16(%A: memref<16x32xi8>, %B: memref<32x16xi8>, %C: memref<16x16xi32>) {
  linalg.matmul ins(%A, %B : memref<16x32xi8>, memref<32x16xi8>)
                outs(%C : memref<16x16xi32>)
  return
}

// Test case 3: int16 matmul (tile size 4x4x4)
// (4x4) * (4x4) = (4x4)
// CHECK-LABEL: func.func @matmul_i16_4x4x4
// CHECK: scf.for
// CHECK:   scf.for
// CHECK:     scf.for
// CHECK:       memref.subview {{.*}} [4, 4]
// CHECK:       memref.subview {{.*}} [4, 4]
// CHECK:       memref.subview {{.*}} [4, 4]
// CHECK:       ime.vmadot
func.func @matmul_i16_4x4x4(%A: memref<4x4xi16>, %B: memref<4x4xi16>, %C: memref<4x4xi32>) {
  linalg.matmul ins(%A, %B : memref<4x4xi16>, memref<4x4xi16>)
                outs(%C : memref<4x4xi32>)
  return
}

// Test case 4: Larger int16 matmul requiring tiling
// (16x16) * (16x16) = (16x16)
// CHECK-LABEL: func.func @matmul_i16_16x16x16
// CHECK: scf.for
// CHECK:   scf.for
// CHECK:     scf.for
// CHECK:       memref.subview {{.*}} [4, 4]
// CHECK:       memref.subview {{.*}} [4, 4]
// CHECK:       memref.subview {{.*}} [4, 4]
// CHECK:       ime.vmadot
func.func @matmul_i16_16x16x16(%A: memref<16x16xi16>, %B: memref<16x16xi16>, %C: memref<16x16xi32>) {
  linalg.matmul ins(%A, %B : memref<16x16xi16>, memref<16x16xi16>)
                outs(%C : memref<16x16xi32>)
  return
}
