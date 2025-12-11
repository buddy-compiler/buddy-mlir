// RUN: buddy-opt %s | FileCheck %s

// This example demonstrates the vmadotu operation for unsigned integer matrix multiplication.
// vmadotu performs: C += A × B where A, B are unsigned int8 matrices and C is int32 accumulator.
//
// Matrix dimensions:
// A: 4×8 (M×K) - unsigned int8
// B: 8×4 (K×N) - unsigned int8
// C: 4×4 (M×N) - int32 accumulator

memref.global "private" @matA : memref<4x8xui8> = dense<[
  [1, 2, 3, 4, 5, 6, 7, 8],
  [1, 2, 3, 4, 5, 6, 7, 8],
  [1, 2, 3, 4, 5, 6, 7, 8],
  [1, 2, 3, 4, 5, 6, 7, 8]
]>

memref.global "private" @matB : memref<8x4xui8> = dense<[
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
  %a = memref.get_global @matA : memref<4x8xui8>
  %b = memref.get_global @matB : memref<8x4xui8>
  
  // Allocate output matrix (accumulator)
  %c = memref.alloc() : memref<4x4xi32>
  
  // Initialize accumulator to zero
  %zero = arith.constant 0 : i32
  linalg.fill ins(%zero : i32) outs(%c : memref<4x4xi32>)
  
  // Perform unsigned × unsigned matrix multiply-accumulate using IME
  // CHECK: ime.vmadotu
  ime.vmadotu %c, %a, %b : memref<4x4xi32>, memref<4x8xui8>, memref<8x4xui8>
  
  // Return success
  %ret = arith.constant 0 : i32
  return %ret : i32
}
