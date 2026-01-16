// RUN: buddy-opt %s | FileCheck %s
// CHECK: func.func @main
//
// vfmadot computes: C[i,j] += sum_k(A[i,k] * B[j,k]) for fp16 values
//
// A (4x8): fp16 values
// B (4x8): fp16 values in packed form
// 
// A row = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
// B row 0,2 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] -> dot = 36.0
// B row 1,3 = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5] -> dot = 18.0

memref.global "private" @matA : memref<4x8xf16> = dense<[
  [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
  [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
  [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
  [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
]>

// Packed B (4x8): fp16 values
memref.global "private" @matB : memref<4x8xf16> = dense<[
  [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
  [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
  [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
  [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
]>

// Expected: C = [[36.0, 18.0, 36.0, 18.0], ...]

func.func private @print_f16_row(i32, f16, f16, f16, f16)
func.func private @print_header()

func.func @main() -> i32 {
  %a = memref.get_global @matA : memref<4x8xf16>
  %b = memref.get_global @matB : memref<4x8xf16>
  
  %c = memref.alloc() : memref<4x4xf16>
  
  // Initialize to zero
  %zero = arith.constant 0.0 : f16
  linalg.fill ins(%zero : f16) outs(%c : memref<4x4xf16>)
  
  // Perform vfmadot (floating-point)
  ime.vfmadot %c, %a, %b : memref<4x4xf16>, memref<4x8xf16>, memref<4x8xf16>
  
  // Print results
  call @print_header() : () -> ()
  
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %i0 = arith.constant 0 : i32
  %i1 = arith.constant 1 : i32
  %i2 = arith.constant 2 : i32
  %i3 = arith.constant 3 : i32
  
  // Row 0
  %v00 = memref.load %c[%c0, %c0] : memref<4x4xf16>
  %v01 = memref.load %c[%c0, %c1] : memref<4x4xf16>
  %v02 = memref.load %c[%c0, %c2] : memref<4x4xf16>
  %v03 = memref.load %c[%c0, %c3] : memref<4x4xf16>
  call @print_f16_row(%i0, %v00, %v01, %v02, %v03) : (i32, f16, f16, f16, f16) -> ()
  
  // Row 1
  %v10 = memref.load %c[%c1, %c0] : memref<4x4xf16>
  %v11 = memref.load %c[%c1, %c1] : memref<4x4xf16>
  %v12 = memref.load %c[%c1, %c2] : memref<4x4xf16>
  %v13 = memref.load %c[%c1, %c3] : memref<4x4xf16>
  call @print_f16_row(%i1, %v10, %v11, %v12, %v13) : (i32, f16, f16, f16, f16) -> ()
  
  // Row 2
  %v20 = memref.load %c[%c2, %c0] : memref<4x4xf16>
  %v21 = memref.load %c[%c2, %c1] : memref<4x4xf16>
  %v22 = memref.load %c[%c2, %c2] : memref<4x4xf16>
  %v23 = memref.load %c[%c2, %c3] : memref<4x4xf16>
  call @print_f16_row(%i2, %v20, %v21, %v22, %v23) : (i32, f16, f16, f16, f16) -> ()
  
  // Row 3
  %v30 = memref.load %c[%c3, %c0] : memref<4x4xf16>
  %v31 = memref.load %c[%c3, %c1] : memref<4x4xf16>
  %v32 = memref.load %c[%c3, %c2] : memref<4x4xf16>
  %v33 = memref.load %c[%c3, %c3] : memref<4x4xf16>
  call @print_f16_row(%i3, %v30, %v31, %v32, %v33) : (i32, f16, f16, f16, f16) -> ()
  
  %ret = arith.constant 0 : i32
  return %ret : i32
}
