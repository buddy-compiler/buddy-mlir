// RUN: buddy-opt %s \
// RUN:     -lower-buckyball | \
// RUN: FileCheck %s

// CHECK: bb_mset
// CHECK: bb_mvout
// CHECK: bb_mset
// CHECK: bb_mset
// CHECK: bb_mvout
// CHECK: bb_mset
// CHECK: bb_mset
// CHECK: bb_mset
// CHECK: bb_mset
// CHECK: bb_mvin
// CHECK: bb_mvin
// CHECK: bb_mvout
// CHECK: bb_mset
// CHECK: bb_mset
// CHECK: bb_mset

// mset(alloc=true) → mvout → mset(alloc=false).

// Fixed parameter version
func.func @main() -> i8 {
  %0 = arith.constant 0 : i8

  // Create output matrix (16-byte lines with default bb_mset lowering)
  %output = memref.alloc() : memref<16x16xi8>

  // Virtual bank id for mvout source
  %bank = arith.constant 0 : i64

  "buckyball.bb_mset"(%bank) : (i64) -> ()
  "buckyball.bb_mvout"(%output, %bank) : (memref<16x16xi8>, i64) -> ()
  "buckyball.bb_mset"(%bank) {alloc = false} : (i64) -> ()

  // Free memory
  memref.dealloc %output : memref<16x16xi8>

  return %0 : i8
}

