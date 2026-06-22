// RUN: buddy-opt %s -lower-buckyball | FileCheck %s
// Im2col test using Bank SSA operations
// CHECK: bb_mset
// CHECK: bb_mvin
// CHECK: bb_im2col
// CHECK: bb_mvout
// CHECK: bb_mset

func.func private @bb_test_report(i32) -> ()

func.func @main() -> i8 {
  %zero = arith.constant 0 : i8
  %input = memref.alloc() : memref<32x16xi8>   // 32 rows, 16 cols (1 line)
  %output = memref.alloc() : memref<52x16xi8>  // im2col output

  // Initialize: input[i,j] = i
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %n_row = arith.constant 32 : index
  %n_col = arith.constant 16 : index
  scf.for %i = %c0 to %n_row step %c1 {
    scf.for %j = %c0 to %n_col step %c1 {
      %idx = arith.index_cast %i : index to i32
      %val_i8 = arith.trunci %idx : i32 to i8
      memref.store %val_i8, %input[%i, %j] : memref<32x16xi8>
    }
  }

  // Bank SSA operations
  %bank_in = "buckyball.bank_alloc"() : () -> i64
  %bank_out = "buckyball.bank_alloc"() : () -> i64
  %depth_in = arith.constant 32 : i64   // 32 rows
  %depth_out = arith.constant 52 : i64  // output rows
  %stride = arith.constant 1 : i64

  // Im2col parameters: krow=4, kcol=1, inrow=16, incol=16, startrow=0, startcol=0
  %krow = arith.constant 4 : i64
  %kcol = arith.constant 1 : i64
  %inrow = arith.constant 16 : i64
  %incol = arith.constant 16 : i64
  %startrow = arith.constant 0 : i64
  %startcol = arith.constant 0 : i64

  %bank_in_loaded = "buckyball.bank_mvin"(%input, %bank_in, %depth_in, %stride)
    : (memref<32x16xi8>, i64, i64, i64) -> i64

  // Im2colOp signature: (inputBankId, outputBankId, krow, kcol, inrow, incol, startrow, startcol)
  %col_step = arith.constant 1 : i64
  "buckyball.im2col"(%bank_in_loaded, %bank_out, %krow, %kcol, %inrow, %incol, %startrow, %startcol, %col_step)
    : (i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()

  %bank_out_stored = "buckyball.bank_mvout"(%output, %bank_out, %depth_out, %stride)
    : (memref<52x16xi8>, i64, i64, i64) -> i64

  "buckyball.bank_release"(%bank_in_loaded) : (i64) -> ()
  "buckyball.bank_release"(%bank_out_stored) : (i64) -> ()

  // Simple verification
  %zero_i32 = arith.constant 0 : i32
  func.call @bb_test_report(%zero_i32) : (i32) -> ()

  memref.dealloc %input : memref<32x16xi8>
  memref.dealloc %output : memref<52x16xi8>
  return %zero : i8
}
