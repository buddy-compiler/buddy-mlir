// RUN: buddy-opt %s | FileCheck %s

// This example demonstrates all integer vmadot variants for different signedness.
//
// vmadot:   signed × signed → signed
// vmadotu:  unsigned × unsigned → unsigned  
// vmadotsu: signed × unsigned → signed
// vmadotus: unsigned × signed → signed

memref.global "private" @matA_signed : memref<4x8xi8> = dense<[
  [1, -2, 3, -4, 5, -6, 7, -8],
  [1, -2, 3, -4, 5, -6, 7, -8],
  [1, -2, 3, -4, 5, -6, 7, -8],
  [1, -2, 3, -4, 5, -6, 7, -8]
]>

memref.global "private" @matB_signed : memref<8x4xi8> = dense<[
  [1, -1, 1, -1],
  [1, -1, 1, -1],
  [1, -1, 1, -1],
  [1, -1, 1, -1],
  [1, -1, 1, -1],
  [1, -1, 1, -1],
  [1, -1, 1, -1],
  [1, -1, 1, -1]
]>

func.func @main() -> i32 {
  %a_signed = memref.get_global @matA_signed : memref<4x8xi8>
  %b_signed = memref.get_global @matB_signed : memref<8x4xi8>
  
  // Allocate accumulators
  %c1 = memref.alloc() : memref<4x4xi32>
  %c2 = memref.alloc() : memref<4x4xi32>
  %c3 = memref.alloc() : memref<4x4xi32>
  %c4 = memref.alloc() : memref<4x4xi32>
  
  // Initialize all accumulators to zero
  %zero = arith.constant 0 : i32
  linalg.fill ins(%zero : i32) outs(%c1 : memref<4x4xi32>)
  linalg.fill ins(%zero : i32) outs(%c2 : memref<4x4xi32>)
  linalg.fill ins(%zero : i32) outs(%c3 : memref<4x4xi32>)
  linalg.fill ins(%zero : i32) outs(%c4 : memref<4x4xi32>)
  
  // vmadot: signed × signed
  // CHECK: ime.vmadot
  ime.vmadot %c1, %a_signed, %b_signed : memref<4x4xi32>, memref<4x8xi8>, memref<8x4xi8>
  
  // vmadotu: unsigned × unsigned (same bit pattern, interpreted as unsigned)
  // CHECK: ime.vmadotu
  ime.vmadotu %c2, %a_signed, %b_signed : memref<4x4xi32>, memref<4x8xi8>, memref<8x4xi8>
  
  // vmadotsu: signed × unsigned  
  // CHECK: ime.vmadotsu
  ime.vmadotsu %c3, %a_signed, %b_signed : memref<4x4xi32>, memref<4x8xi8>, memref<8x4xi8>
  
  // vmadotus: unsigned × signed
  // CHECK: ime.vmadotus
  ime.vmadotus %c4, %a_signed, %b_signed : memref<4x4xi32>, memref<4x8xi8>, memref<8x4xi8>
  
  %ret = arith.constant 0 : i32
  return %ret : i32
}
