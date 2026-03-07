// RUN: buddy-opt %s \
// RUN:     -lower-buckyball | \
// RUN: FileCheck %s

// Fixed parameter version
func.func @main() -> i8 {
  %0 = arith.constant 0 : i8
  
  // Create output matrix
  %output = memref.alloc() : memref<16x32xi8>
  
  // Get scratchpad address (scratchpad address)
  %sp_addr = arith.constant 100 : i64  // Scratchpad address
  
  // Use Buckyball's mvout operation to move data from scratchpad to memory
  // CHECK: mvout
  "buckyball.mvout"(%output, %sp_addr) : (memref<16x32xi8>, i64) -> ()
  
  // Free memory
  memref.dealloc %output : memref<16x32xi8>
  
  return %0 : i8
}

// Dynamic parameter version
func.func @dynamic_test(%rows: index, %cols: index, %addr: i64) -> i8 {
  %0 = arith.constant 0 : i8
  
  // Create dynamic size output matrix
  %output = memref.alloc(%rows, %cols) : memref<?x?xi8>
  
  // Use dynamic parameter mvout operation
  // CHECK: mvout
  "buckyball.mvout"(%output, %addr) : (memref<?x?xi8>, i64) -> ()
  
  // Free memory
  memref.dealloc %output : memref<?x?xi8>
  
  return %0 : i8
} 

// Example of usage in actual calculation scenario
func.func @matmul_test(%m: index, %n: index, %k: index) -> i8 {
  %0 = arith.constant 0 : i8
  
  // Create matrix
  %a = memref.alloc(%m, %k) : memref<?x?xi8>
  %b = memref.alloc(%k, %n) : memref<?x?xi8>
  %c = memref.alloc(%m, %n) : memref<?x?xi8>
  
  // Scratchpad address
  %a_addr = arith.constant 0 : i64
  %b_addr = arith.constant 1000 : i64
  %c_addr = arith.constant 2000 : i64
  
  // Load A, B matrices to scratchpad
  "buckyball.mvin"(%a, %a_addr) : (memref<?x?xi8>, i64) -> ()
  "buckyball.mvin"(%b, %b_addr) : (memref<?x?xi8>, i64) -> ()
  
  // Perform matrix multiplication (omitted here)
  
  // Write results back to memory from scratchpad
  // CHECK: mvout
  "buckyball.mvout"(%c, %c_addr) : (memref<?x?xi8>, i64) -> ()
  
  // Free memory
  memref.dealloc %a : memref<?x?xi8>
  memref.dealloc %b : memref<?x?xi8>
  memref.dealloc %c : memref<?x?xi8>
  
  return %0 : i8
} 