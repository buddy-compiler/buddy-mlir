// RUN: buddy-opt %s -lower-buckyball | FileCheck %s
// Transpose test using Bank SSA operations
// CHECK: bb_mset
// CHECK: bb_mvin
// CHECK: bb_transpose
// CHECK: bb_mvout
// CHECK: bb_mset

func.func private @bb_test_report(i32) -> ()

func.func @main() -> i8 {
  %zero = arith.constant 0 : i8
  %input = memref.alloc() : memref<16x16xi8>
  %output = memref.alloc() : memref<16x16xi8>

  // Initialize: input[i,j] = i * 16 + j
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %n = arith.constant 16 : index
  %c16_i32 = arith.constant 16 : i32
  scf.for %i = %c0 to %n step %c1 {
    scf.for %j = %c0 to %n step %c1 {
      %idx = arith.index_cast %i : index to i32
      %jdx = arith.index_cast %j : index to i32
      %mul = arith.muli %idx, %c16_i32 : i32
      %val = arith.addi %mul, %jdx : i32
      %val_i8 = arith.trunci %val : i32 to i8
      memref.store %val_i8, %input[%i, %j] : memref<16x16xi8>
    }
  }

  // Bank SSA operations
  %bank_in = "buckyball.bank_alloc"() : () -> i64
  %bank_out = "buckyball.bank_alloc"() : () -> i64
  %depth = arith.constant 16 : i64
  %stride = arith.constant 1 : i64
  %iter = arith.constant 16 : i64
  %mode = arith.constant 0 : i64

  %bank_in_loaded = "buckyball.bank_mvin"(%input, %bank_in, %depth, %stride)
    : (memref<16x16xi8>, i64, i64, i64) -> i64

  // TransposeOp signature: (inputBankId, outputBankId, iter, mode)
  "buckyball.transpose"(%bank_in_loaded, %bank_out, %iter, %mode)
    : (i64, i64, i64, i64) -> ()

  %bank_out_stored = "buckyball.bank_mvout"(%output, %bank_out, %depth, %stride)
    : (memref<16x16xi8>, i64, i64, i64) -> i64

  "buckyball.bank_release"(%bank_in_loaded) : (i64) -> ()
  "buckyball.bank_release"(%bank_out_stored) : (i64) -> ()

  // Verify: output[0,1] should equal input[1,0]
  %v_in = memref.load %input[%c1, %c0] : memref<16x16xi8>
  %v_out = memref.load %output[%c0, %c1] : memref<16x16xi8>
  %v_in_i32 = arith.extui %v_in : i8 to i32
  %v_out_i32 = arith.extui %v_out : i8 to i32
  %fail_i1 = arith.cmpi ne, %v_in_i32, %v_out_i32 : i32
  %fail = arith.extui %fail_i1 : i1 to i32
  func.call @bb_test_report(%fail) : (i32) -> ()

  memref.dealloc %input : memref<16x16xi8>
  memref.dealloc %output : memref<16x16xi8>
  return %zero : i8
}
