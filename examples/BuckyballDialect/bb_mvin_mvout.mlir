// RUN: buddy-opt %s \
// RUN:     -lower-buckyball | \
// RUN: FileCheck %s

// One bank session: mvin then mvout. `depth` / `stride` match isa.h + 33_mvin.c / 16_mvout.c
// (BB_ITER(depth), stride in rs2). For MxK i8 with 16-byte lines: depth = M*(K/16).
// Host: debug_helper.c -> bb_test_report.
// CHECK: bb_mset
// CHECK: bb_mvin
// CHECK: bb_mvout
// CHECK: bb_mset

"memref.global"() {sym_name = "input_a", type = memref<16x1024xi8>, initial_value = dense<3> : tensor<16x1024xi8>, visibility = "private"} : () -> ()

func.func private @bb_test_report(i32) -> ()

func.func @main() -> i8 {
  %zero = arith.constant 0 : i8
  %one = arith.constant 1 : i8
  %a = memref.get_global @input_a : memref<16x1024xi8>
  %b = memref.alloc() : memref<16x1024xi8>
  %bank = arith.constant 0 : i64
  // 16 rows * (1024/16) lines per row = 1024 (same convention as bb_matmul lowering depthA)
  %depth = arith.constant 1024 : i64
  %stride = arith.constant 1 : i64

  "buckyball.bb_mset"(%bank) : (i64) -> ()
  "buckyball.bb_mvin"(%a, %bank, %depth, %stride) : (memref<16x1024xi8>, i64, i64, i64) -> ()
  "buckyball.bb_mvout"(%b, %bank, %depth, %stride) : (memref<16x1024xi8>, i64, i64, i64) -> ()
  "buckyball.bb_mset"(%bank) {alloc = false} : (i64) -> ()

  %c0 = arith.constant 0 : index
  %va = memref.load %a[%c0, %c0] : memref<16x1024xi8>
  %vb = memref.load %b[%c0, %c0] : memref<16x1024xi8>
  %bad = arith.cmpi ne, %va, %vb : i8
  %fail = arith.extui %bad : i1 to i32
  func.call @bb_test_report(%fail) : (i32) -> ()
  %rc = arith.select %bad, %one, %zero : i8

  memref.dealloc %b : memref<16x1024xi8>
  return %rc : i8
}
