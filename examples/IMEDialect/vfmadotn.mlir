// RUN: buddy-opt %s | FileCheck %s

// This example demonstrates the IME vfmadotn operation with dynamic slide parameter.
// vfmadotn performs: C += slide(A, n) × B where A, B are fp16 matrices.
//
// Sliding window: reads from VS1 and VS1+1 (64 elements), slides by n rows.
// Matrix dimensions for VLEN=256, SEW=16 (fp16):
// A: 8×4 (2M×K) source, sliding selects 4×4 (M×K) - fp16
// B: 4×4 (K×N) - fp16
// C: 4×4 (M×N) - fp16 accumulator

memref.global "private" @matA : memref<8x4xf16> = dense<[
  [1.0, 2.0, 3.0, 4.0],
  [2.0, 3.0, 4.0, 5.0],
  [3.0, 4.0, 5.0, 6.0],
  [4.0, 5.0, 6.0, 7.0],
  [5.0, 6.0, 7.0, 8.0],
  [6.0, 7.0, 8.0, 9.0],
  [7.0, 8.0, 9.0, 10.0],
  [8.0, 9.0, 10.0, 11.0]
]>

memref.global "private" @matB : memref<4x4xf16> = dense<[
  [1.0, 1.0, 1.0, 1.0],
  [1.0, 1.0, 1.0, 1.0],
  [1.0, 1.0, 1.0, 1.0],
  [1.0, 1.0, 1.0, 1.0]
]>

func.func @main() -> i32 {
  // Get input matrices
  %a = memref.get_global @matA : memref<8x4xf16>
  %b = memref.get_global @matB : memref<4x4xf16>
  
  // Allocate output matrix (accumulator)
  %c = memref.alloc() : memref<4x4xf16>
  
  // Initialize accumulator to zero
  %zero = arith.constant 0.0 : f16
  linalg.fill ins(%zero : f16) outs(%c : memref<4x4xf16>)
  
  // Slide parameter (0-3)
  %slide = arith.constant 1 : i64
  
  // Perform floating-point matrix multiply-accumulate with dynamic slide
  // CHECK: ime.vfmadotn
  ime.vfmadotn %c, %a, %b, %slide : memref<4x4xf16>, memref<8x4xf16>, memref<4x4xf16>
  
  // Return success
  %ret = arith.constant 0 : i32
  return %ret : i32
}
