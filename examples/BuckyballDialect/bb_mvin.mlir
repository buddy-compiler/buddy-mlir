// RUN: buddy-opt %s \
// RUN:     -lower-buckyball | \
// RUN: FileCheck %s

// Fixed parameter version
func.func @main() -> i8 {
  %0 = arith.constant 0 : i8
  
  // Create input matrix
  %input = memref.alloc() : memref<16x32xi8>
  
  // Get scratchpad address (scratchpad address)
  %sp_addr = arith.constant 100 : i64  // Scratchpad address
  
  // Use Buckyball's mvin operation to move data from memory to scratchpad
  // CHECK: mvin
  "buckyball.mvin"(%input, %sp_addr) : (memref<16x32xi8>, i64) -> ()
  
  // Free memory
  memref.dealloc %input : memref<16x32xi8>
  
  return %0 : i8
}

// Dynamic parameter version
func.func @dynamic_test(%rows: index, %cols: index, %addr: i64) -> i8 {
  %0 = arith.constant 0 : i8
  
  // Create dynamic size input matrix
  %input = memref.alloc(%rows, %cols) : memref<?x?xi8>
  
  // Use dynamic parameter mvin operation
  // CHECK: mvin
  "buckyball.mvin"(%input, %addr) : (memref<?x?xi8>, i64) -> ()
  
  // Free memory
  memref.dealloc %input : memref<?x?xi8>
  
  return %0 : i8
} 