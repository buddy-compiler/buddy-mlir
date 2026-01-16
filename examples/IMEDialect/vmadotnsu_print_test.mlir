// RUN: buddy-opt %s | FileCheck %s
// CHECK: func.func @main
//
// vmadotnsu computes: C[i,j] += sum_k(signed(A[slide+i,k]) * unsigned(B[j,k]))
//
// Sliding window reads 64 elements from VS1 (8 rows), then slides by n rows.
// A (8x8): signed int8, source matrix (negative values)
// B (4x8): unsigned int8, packed form
//
// With slide=1:
// A rows used = [1,2,3,4] (after sliding by 1)
// Row 1 = [-2,-3,-4,-5,-6,-7,-8,-9], sum = -44
// Row 2 = [-3,-4,-5,-6,-7,-8,-9,-10], sum = -52
// Row 3 = [-4,-5,-6,-7,-8,-9,-10,-11], sum = -60
// Row 4 = [-5,-6,-7,-8,-9,-10,-11,-12], sum = -68

memref.global "private" @matA : memref<8x8xi8> = dense<[
  [-1, -2, -3, -4, -5, -6, -7, -8],
  [-2, -3, -4, -5, -6, -7, -8, -9],
  [-3, -4, -5, -6, -7, -8, -9, -10],
  [-4, -5, -6, -7, -8, -9, -10, -11],
  [-5, -6, -7, -8, -9, -10, -11, -12],
  [-6, -7, -8, -9, -10, -11, -12, -13],
  [-7, -8, -9, -10, -11, -12, -13, -14],
  [-8, -9, -10, -11, -12, -13, -14, -15]
]>

// Packed B (4x8): all ones (unsigned)
memref.global "private" @matB : memref<4x8xi8> = dense<[
  [1, 1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1, 1]
]>

// With slide=1: Expected C = [[-44,-44,-44,-44], [-52,-52,-52,-52], [-60,-60,-60,-60], [-68,-68,-68,-68]]

func.func private @print_row(i32, i32, i32, i32, i32)
func.func private @print_header()

func.func @main() -> i32 {
  %a = memref.get_global @matA : memref<8x8xi8>
  %b = memref.get_global @matB : memref<4x8xi8>
  
  %c = memref.alloc() : memref<4x4xi32>
  
  // Initialize to zero
  %zero = arith.constant 0 : i32
  linalg.fill ins(%zero : i32) outs(%c : memref<4x4xi32>)
  
  // Slide parameter = 1
  %slide = arith.constant 1 : i64
  
  // Perform vmadotnsu with slide=1
  ime.vmadotnsu %c, %a, %b, %slide : memref<4x4xi32>, memref<8x8xi8>, memref<4x8xi8>
  
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
  %v00 = memref.load %c[%c0, %c0] : memref<4x4xi32>
  %v01 = memref.load %c[%c0, %c1] : memref<4x4xi32>
  %v02 = memref.load %c[%c0, %c2] : memref<4x4xi32>
  %v03 = memref.load %c[%c0, %c3] : memref<4x4xi32>
  call @print_row(%i0, %v00, %v01, %v02, %v03) : (i32, i32, i32, i32, i32) -> ()
  
  // Row 1
  %v10 = memref.load %c[%c1, %c0] : memref<4x4xi32>
  %v11 = memref.load %c[%c1, %c1] : memref<4x4xi32>
  %v12 = memref.load %c[%c1, %c2] : memref<4x4xi32>
  %v13 = memref.load %c[%c1, %c3] : memref<4x4xi32>
  call @print_row(%i1, %v10, %v11, %v12, %v13) : (i32, i32, i32, i32, i32) -> ()
  
  // Row 2
  %v20 = memref.load %c[%c2, %c0] : memref<4x4xi32>
  %v21 = memref.load %c[%c2, %c1] : memref<4x4xi32>
  %v22 = memref.load %c[%c2, %c2] : memref<4x4xi32>
  %v23 = memref.load %c[%c2, %c3] : memref<4x4xi32>
  call @print_row(%i2, %v20, %v21, %v22, %v23) : (i32, i32, i32, i32, i32) -> ()
  
  // Row 3
  %v30 = memref.load %c[%c3, %c0] : memref<4x4xi32>
  %v31 = memref.load %c[%c3, %c1] : memref<4x4xi32>
  %v32 = memref.load %c[%c3, %c2] : memref<4x4xi32>
  %v33 = memref.load %c[%c3, %c3] : memref<4x4xi32>
  call @print_row(%i3, %v30, %v31, %v32, %v33) : (i32, i32, i32, i32, i32) -> ()
  
  %ret = arith.constant 0 : i32
  return %ret : i32
}
