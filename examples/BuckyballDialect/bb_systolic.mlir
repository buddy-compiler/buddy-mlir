// RUN: buddy-opt %s -lower-buckyball | FileCheck %s
// Simple Systolic Array test using Bank SSA operations
// CHECK: bb_mset
// CHECK: bb_mvin
// CHECK: bb_bbfp_mul
// CHECK: bb_mvout
// CHECK: bb_mset

func.func private @bb_test_report(i32) -> ()

func.func @main() -> i8 {
  %zero = arith.constant 0 : i8
  %a = memref.alloc() : memref<16x16xi8>
  %b = memref.alloc() : memref<16x16xi8>
  %c = memref.alloc() : memref<16x16xi32>

  // Initialize: a[i,j] = 1, b[i,j] = 1
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %n = arith.constant 16 : index
  %one_i8 = arith.constant 1 : i8
  scf.for %i = %c0 to %n step %c1 {
    scf.for %j = %c0 to %n step %c1 {
      memref.store %one_i8, %a[%i, %j] : memref<16x16xi8>
      memref.store %one_i8, %b[%i, %j] : memref<16x16xi8>
    }
  }

  // Bank SSA operations
  %bank_a = "buckyball.bank_alloc"() : () -> i64
  %bank_b = "buckyball.bank_alloc"() : () -> i64
  %bank_c = "buckyball.bank_alloc"() : () -> i64
  %depth = arith.constant 16 : i64
  %stride = arith.constant 1 : i64
  %config = arith.constant 0 : i64

  %bank_a_loaded = "buckyball.bank_mvin"(%a, %bank_a, %depth, %stride)
    : (memref<16x16xi8>, i64, i64, i64) -> i64
  %bank_b_loaded = "buckyball.bank_mvin"(%b, %bank_b, %depth, %stride)
    : (memref<16x16xi8>, i64, i64, i64) -> i64

  // Note: SystolicOp signature is (op1BankId, op2BankId, resultBankId, config)
  "buckyball.systolic"(%bank_a_loaded, %bank_b_loaded, %bank_c, %config)
    : (i64, i64, i64, i64) -> ()

  %bank_c_stored = "buckyball.bank_mvout"(%c, %bank_c, %depth, %stride)
    : (memref<16x16xi32>, i64, i64, i64) -> i64

  "buckyball.bank_release"(%bank_a_loaded) : (i64) -> ()
  "buckyball.bank_release"(%bank_b_loaded) : (i64) -> ()
  "buckyball.bank_release"(%bank_c_stored) : (i64) -> ()

  %zero_i32 = arith.constant 0 : i32
  func.call @bb_test_report(%zero_i32) : (i32) -> ()

  memref.dealloc %a : memref<16x16xi8>
  memref.dealloc %b : memref<16x16xi8>
  memref.dealloc %c : memref<16x16xi32>
  return %zero : i8
}
