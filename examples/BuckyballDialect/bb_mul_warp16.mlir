// RUN: buddy-opt %s -lower-buckyball | FileCheck %s
// MulWarp16 test using Bank SSA operations
// CHECK: bb_mset
// CHECK: bb_mvin
// CHECK: bb_mul_warp16
// CHECK: bb_mvout
// CHECK: bb_mset

func.func private @bb_test_report(i32) -> ()

func.func @main() -> i8 {
  %zero = arith.constant 0 : i8
  %a = memref.alloc() : memref<16x1024xi8>
  %b = memref.alloc() : memref<16x1024xi8>
  %c = memref.alloc() : memref<16x16xi32>

  // Initialize: a[i,j] = 1, b[i,j] = 1
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %n_row = arith.constant 16 : index
  %n_col = arith.constant 1024 : index
  %one_i8 = arith.constant 1 : i8
  scf.for %i = %c0 to %n_row step %c1 {
    scf.for %j = %c0 to %n_col step %c1 {
      memref.store %one_i8, %a[%i, %j] : memref<16x1024xi8>
      memref.store %one_i8, %b[%i, %j] : memref<16x1024xi8>
    }
  }

  // Bank SSA operations
  %bank_a = "buckyball.bank_alloc"() : () -> i64
  %bank_b = "buckyball.bank_alloc"() : () -> i64
  %bank_c = "buckyball.bank_alloc"() : () -> i64
  // 16 rows * 1024 cols / 16 bytes per line
  %depth_in = arith.constant 1024 : i64
  // 16x16 result
  %depth_out = arith.constant 16 : i64
  %stride = arith.constant 1 : i64
  %iter = arith.constant 1024 : i64
  %mode = arith.constant 0 : i64

  %bank_a_loaded = "buckyball.bank_mvin"(%a, %bank_a, %depth_in, %stride)
    : (memref<16x1024xi8>, i64, i64, i64) -> i64
  %bank_b_loaded = "buckyball.bank_mvin"(%b, %bank_b, %depth_in, %stride)
    : (memref<16x1024xi8>, i64, i64, i64) -> i64

  // MulWarp16Op signature: (op1BankId, op2BankId, wrBankId, iter, mode)
  "buckyball.mul_warp16"(%bank_a_loaded, %bank_b_loaded, %bank_c, %iter, %mode)
    : (i64, i64, i64, i64, i64) -> ()

  %bank_c_stored = "buckyball.bank_mvout"(%c, %bank_c, %depth_out, %stride)
    : (memref<16x16xi32>, i64, i64, i64) -> i64

  "buckyball.bank_release"(%bank_a_loaded) : (i64) -> ()
  "buckyball.bank_release"(%bank_b_loaded) : (i64) -> ()
  "buckyball.bank_release"(%bank_c_stored) : (i64) -> ()

  // Verify: c[0,0] should be 1024
  // (sum of 1024 multiplications of 1*1)
  %expected = arith.constant 1024 : i32
  %result = memref.load %c[%c0, %c0] : memref<16x16xi32>
  %fail_i1 = arith.cmpi ne, %result, %expected : i32
  %fail = arith.extui %fail_i1 : i1 to i32
  func.call @bb_test_report(%fail) : (i32) -> ()

  memref.dealloc %a : memref<16x1024xi8>
  memref.dealloc %b : memref<16x1024xi8>
  memref.dealloc %c : memref<16x16xi32>
  return %zero : i8
}
