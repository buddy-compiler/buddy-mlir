// RUN: buddy-opt %s \
// RUN:     -lower-buckyball | \
// RUN: FileCheck %s

// There is no standalone bb_mul_warp16 op in the Buckyball dialect; bb_matmul lowers to
// mset/mvin/mul_warp16/mvout (see bb-tests OpTest toy/bb_mul_warp16.mlir).

// Fixed parameter version
func.func @main() -> i8 {
  %0 = arith.constant 0 : i8

  %a = memref.alloc() : memref<16x16xi8>
  %b = memref.alloc() : memref<16x16xi8>
  %c = memref.alloc() : memref<16x16xi8>

  // bb_matmul: lowers to mvin A + mvin B + mul_warp16 + mvout C
  // CHECK: bb_mul_warp16
  "buckyball.bb_matmul"(%a, %b, %c) : (memref<16x16xi8>, memref<16x16xi8>, memref<16x16xi8>) -> ()

  memref.dealloc %a : memref<16x16xi8>
  memref.dealloc %b : memref<16x16xi8>
  memref.dealloc %c : memref<16x16xi8>

  return %0 : i8
}
