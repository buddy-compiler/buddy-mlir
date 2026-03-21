// RUN: buddy-opt %s \
// RUN:     -lower-buckyball | \
// RUN: FileCheck %s

// CHECK: bb_mset
// CHECK: bb_mvin
// CHECK: bb_mset
// CHECK: bb_mset
// CHECK: bb_mvin
// CHECK: bb_mset

// bb_mset(alloc=true) → mvin → bb_mset(alloc=false) to release the virtual bank.
// Default lowering uses mset row=1,col=1 → 16-byte lines; use 16x16 i8 rows here.

// Fixed parameter version
func.func @main() -> i8 {
  %0 = arith.constant 0 : i8

  // Create input matrix
  %input = memref.alloc() : memref<16x16xi8>

  // Virtual bank id and line stride for the mvin encoding
  %bank = arith.constant 0 : i64
  %stride = arith.constant 1 : i64

  "buckyball.bb_mset"(%bank) : (i64) -> ()
  "buckyball.bb_mvin"(%input, %bank, %stride) : (memref<16x16xi8>, i64, i64) -> ()
  "buckyball.bb_mset"(%bank) {alloc = false} : (i64) -> ()

  // Free memory
  memref.dealloc %input : memref<16x16xi8>

  return %0 : i8
}
