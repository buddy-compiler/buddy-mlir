// RUN: buddy-opt %s \
// RUN:     -lower-buckyball | \
// RUN: FileCheck %s

// Tests **bb_mul_warp16** only (`64_mul_warp16.c`), not bb_matmul. Sequence:
// mset banks 0,1,2 (bank2 col=4 for wr) → mvin A,B → bb_mul_warp16 → mvout C → mset release.
// A,B: 16×1024 i8 all ones; K=iter=1024 ⇒ C[0,0] == 1024 (16×16 acc tile).
// mvin depth=1024 (cols=1: 1024 lines × 16 B); mvout depth=16 for 16×16 i32 tile.
// Host: debug_helper.c -> bb_test_report.
// CHECK: bb_mul_warp16

"memref.global"() {sym_name = "a_g", type = memref<16x1024xi8>, initial_value = dense<1> : tensor<16x1024xi8>, visibility = "private"} : () -> ()
"memref.global"() {sym_name = "b_g", type = memref<16x1024xi8>, initial_value = dense<1> : tensor<16x1024xi8>, visibility = "private"} : () -> ()

func.func private @bb_test_report(i32) -> ()

func.func @main() -> i8 {
  %zero = arith.constant 0 : i8
  %one = arith.constant 1 : i8
  %exp = arith.constant 1024 : i32

  %a = memref.get_global @a_g : memref<16x1024xi8>
  %b = memref.get_global @b_g : memref<16x1024xi8>
  %c = memref.alloc() : memref<16x16xi32>

  %bk0 = arith.constant 0 : i64
  %bk1 = arith.constant 1 : i64
  %bk2 = arith.constant 2 : i64
  %depthIn = arith.constant 1024 : i64
  %depthOut = arith.constant 16 : i64
  %stride = arith.constant 1 : i64
  %iter = arith.constant 1024 : i64
  %mode0 = arith.constant 0 : i64

  "buckyball.bb_mset"(%bk0) : (i64) -> ()
  "buckyball.bb_mset"(%bk1) : (i64) -> ()
  "buckyball.bb_mset"(%bk2) {col = 4 : i64} : (i64) -> ()

  "buckyball.bb_mvin"(%a, %bk0, %depthIn, %stride) : (memref<16x1024xi8>, i64, i64, i64) -> ()
  "buckyball.bb_mvin"(%b, %bk1, %depthIn, %stride) : (memref<16x1024xi8>, i64, i64, i64) -> ()

  "buckyball.bb_mul_warp16"(%bk0, %bk1, %bk2, %iter, %mode0) : (i64, i64, i64, i64, i64) -> ()

  "buckyball.bb_mvout"(%c, %bk2, %depthOut, %stride) : (memref<16x16xi32>, i64, i64, i64) -> ()

  "buckyball.bb_mset"(%bk0) {alloc = false} : (i64) -> ()
  "buckyball.bb_mset"(%bk1) {alloc = false} : (i64) -> ()
  "buckyball.bb_mset"(%bk2) {alloc = false} : (i64) -> ()

  %i0 = arith.constant 0 : index
  %got = memref.load %c[%i0, %i0] : memref<16x16xi32>
  %bad = arith.cmpi ne, %got, %exp : i32
  %fail = arith.extui %bad : i1 to i32
  func.call @bb_test_report(%fail) : (i32) -> ()
  %rc = arith.select %bad, %one, %zero : i8

  memref.dealloc %c : memref<16x16xi32>
  return %rc : i8
}
