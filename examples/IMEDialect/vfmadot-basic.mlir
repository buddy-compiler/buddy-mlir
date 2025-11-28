// RUN: buddy-opt %s | FileCheck %s

// This example demonstrates the vfmadot operation for fp16 matrix multiplication.
// vfmadot performs: C += A × B where A, B are fp16 matrices and C is fp32 accumulator.
//
// Matrix dimensions for VLEN=256, SEW=16:
// A: 4×4 (M×K)
// B: 4×4 (K×N)  
// C: 4×4 (M×N)

memref.global "private" @matA : memref<4x4xf16> = dense<[
  [1.0, 2.0, 3.0, 4.0],
  [1.0, 2.0, 3.0, 4.0],
  [1.0, 2.0, 3.0, 4.0],
  [1.0, 2.0, 3.0, 4.0]
]>

memref.global "private" @matB : memref<4x4xf16> = dense<[
  [1.0, 1.0, 1.0, 1.0],
  [1.0, 1.0, 1.0, 1.0],
  [1.0, 1.0, 1.0, 1.0],
  [1.0, 1.0, 1.0, 1.0]
]>

func.func @main() -> i32 {
  // Get input matrices
  %a = memref.get_global @matA : memref<4x4xf16>
  %b = memref.get_global @matB : memref<4x4xf16>
  
  // Allocate output matrix (accumulator)
  %c = memref.alloc() : memref<4x4xf32>
  
  // Initialize accumulator to zero
  %zero = arith.constant 0.0 : f32
  linalg.fill ins(%zero : f32) outs(%c : memref<4x4xf32>)
  
  // Perform matrix multiply-accumulate using IME
  // CHECK: ime.vfmadot
  ime.vfmadot %c, %a, %b : memref<4x4xf32>, memref<4x4xf16>, memref<4x4xf16>
  
  // Return success
  %ret = arith.constant 0 : i32
  return %ret : i32
}
