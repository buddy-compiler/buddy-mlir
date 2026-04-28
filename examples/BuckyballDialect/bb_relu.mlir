// RUN: buddy-opt %s -lower-buckyball | FileCheck %s
// Simple ReLU test using Bank SSA operations
// CHECK: bb_mset
// CHECK: bb_mvin
// CHECK: bb_relu
// CHECK: bb_mvout
// CHECK: bb_mset

func.func private @bb_test_report(i32) -> ()

func.func @main() -> i8 {
  %zero = arith.constant 0 : i8
  %input = memref.alloc() : memref<16x16xi32>
  %output = memref.alloc() : memref<16x16xi32>

  // Initialize: input[i,j] = i - j (mixed positive/negative)
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %n = arith.constant 16 : index
  scf.for %i = %c0 to %n step %c1 {
    scf.for %j = %c0 to %n step %c1 {
      %idx = arith.index_cast %i : index to i32
      %jdx = arith.index_cast %j : index to i32
      %val = arith.subi %idx, %jdx : i32
      memref.store %val, %input[%i, %j] : memref<16x16xi32>
    }
  }

  // Bank SSA operations: manual mvin/mvout
  %bank_in = "buckyball.bank_alloc"() : () -> i64
  %bank_out = "buckyball.bank_alloc"() : () -> i64
  %depth = arith.constant 16 : i64
  %stride = arith.constant 1 : i64

  %bank_in_loaded = "buckyball.bank_mvin"(%input, %bank_in, %depth, %stride)
    : (memref<16x16xi32>, i64, i64, i64) -> i64

  // Note: ReluOp signature is (inputBankId, outputBankId, depth, stride)
  "buckyball.relu"(%bank_in_loaded, %bank_out, %depth, %stride)
    : (i64, i64, i64, i64) -> ()

  %bank_out_stored = "buckyball.bank_mvout"(%output, %bank_out, %depth, %stride)
    : (memref<16x16xi32>, i64, i64, i64) -> i64

  "buckyball.bank_release"(%bank_in_loaded) : (i64) -> ()
  "buckyball.bank_release"(%bank_out_stored) : (i64) -> ()

  // Simple check: verify output[0,0] = max(0, 0-0) = 0
  %v = memref.load %output[%c0, %c0] : memref<16x16xi32>
  %zero_i32 = arith.constant 0 : i32
  %fail_i1 = arith.cmpi ne, %v, %zero_i32 : i32
  %fail = arith.extui %fail_i1 : i1 to i32
  func.call @bb_test_report(%fail) : (i32) -> ()

  memref.dealloc %input : memref<16x16xi32>
  memref.dealloc %output : memref<16x16xi32>
  return %zero : i8
}
