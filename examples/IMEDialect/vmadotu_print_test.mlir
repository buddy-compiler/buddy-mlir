// IME vmadotu test: unsigned × unsigned matrix multiply-accumulate
//
// vmadotu computes: C[i,j] += sum_k(unsigned(A[i,k]) * unsigned(B[j,k]))
//
// A (4x8): unsigned int8, positive values
// B (4x8): unsigned int8, packed form, positive values
//
// A row = [1, 2, 3, 4, 5, 6, 7, 8]
// B row 0,2 = [1,1,1,1,1,1,1,1] -> dot = 1+2+3+4+5+6+7+8 = 36
// B row 1,3 = [2,2,2,2,2,2,2,2] -> dot = 2+4+6+8+10+12+14+16 = 72

memref.global "private" @matA : memref<4x8xui8> = dense<[
  [1, 2, 3, 4, 5, 6, 7, 8],
  [1, 2, 3, 4, 5, 6, 7, 8],
  [1, 2, 3, 4, 5, 6, 7, 8],
  [1, 2, 3, 4, 5, 6, 7, 8]
]>

// Packed B (4x8): unsigned values
memref.global "private" @matB : memref<4x8xui8> = dense<[
  [1, 1, 1, 1, 1, 1, 1, 1],
  [2, 2, 2, 2, 2, 2, 2, 2],
  [1, 1, 1, 1, 1, 1, 1, 1],
  [2, 2, 2, 2, 2, 2, 2, 2]
]>

// Expected: C = [[36,72,36,72], [36,72,36,72], [36,72,36,72], [36,72,36,72]]

func.func private @print_row(i32, i32, i32, i32, i32)
func.func private @print_header()

func.func @main() -> i32 {
  %a = memref.get_global @matA : memref<4x8xui8>
  %b = memref.get_global @matB : memref<4x8xui8>
  
  %c = memref.alloc() : memref<4x4xi32>
  
  // Initialize to zero
  %zero = arith.constant 0 : i32
  linalg.fill ins(%zero : i32) outs(%c : memref<4x4xi32>)
  
  // Perform vmadotu (unsigned × unsigned)
  ime.vmadotu %c, %a, %b : memref<4x4xi32>, memref<4x8xui8>, memref<4x8xui8>
  
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
