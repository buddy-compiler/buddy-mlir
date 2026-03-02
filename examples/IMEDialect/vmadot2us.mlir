// RUN: buddy-opt %s | FileCheck %s

// This example demonstrates the IME vmadot2us operation (slide=2, unsigned × signed).
// vmadot2us performs: C += slide(A, 2) × B where A is unsigned, B is signed int8 matrices.
//
// Sliding window: reads from VS1 and VS1+1 (64 elements), slides by 2 rows.
// Matrix dimensions for VLEN=256, SEW=8:
// A: 8×8 (2M×K) source, sliding selects 4×8 (M×K) starting from row 2 - unsigned int8
// B: 8×4 (K×N) - signed int8
// C: 4×4 (M×N) - int32 accumulator

memref.global "private" @matA : memref<8x8xui8> = dense<[
  [1, 2, 3, 4, 5, 6, 7, 8],
  [2, 3, 4, 5, 6, 7, 8, 9],
  [3, 4, 5, 6, 7, 8, 9, 10],
  [4, 5, 6, 7, 8, 9, 10, 11],
  [5, 6, 7, 8, 9, 10, 11, 12],
  [6, 7, 8, 9, 10, 11, 12, 13],
  [7, 8, 9, 10, 11, 12, 13, 14],
  [8, 9, 10, 11, 12, 13, 14, 15]
]>

memref.global "private" @matB : memref<8x4xi8> = dense<[
  [1, 1, 1, 1],
  [1, 1, 1, 1],
  [1, 1, 1, 1],
  [1, 1, 1, 1],
  [1, 1, 1, 1],
  [1, 1, 1, 1],
  [1, 1, 1, 1],
  [1, 1, 1, 1]
]>

func.func @main() -> i32 {
  // Get input matrices
  %a = memref.get_global @matA : memref<8x8xui8>
  %b = memref.get_global @matB : memref<8x4xi8>
  
  // Allocate output matrix (accumulator)
  %c = memref.alloc() : memref<4x4xi32>
  
  // Initialize accumulator to zero
  %zero = arith.constant 0 : i32
  linalg.fill ins(%zero : i32) outs(%c : memref<4x4xi32>)
  
  // Perform unsigned × signed matrix multiply-accumulate with slide=2
  // CHECK: ime.vmadot2us
  ime.vmadot2us %c, %a, %b : memref<4x4xi32>, memref<8x8xui8>, memref<8x4xi8>
  
  // Return success
  %ret = arith.constant 0 : i32
  return %ret : i32
}
