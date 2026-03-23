// RUN: buddy-opt %s \
// RUN:     -lower-buckyball | \
// RUN: FileCheck %s

// DUT: single `bb_im2col` (lowers to mset / mvin / bb_im2col / mvout / mset release).
// GRM: pure MLIR, layout matches toy im2col_test.c:
//   lin = window_idx * (krow*kcol) + elem_idx, then reshape [52,16].
// CHECK: bb_mset
// CHECK: bb_mvin
// CHECK: bb_im2col
// CHECK: bb_mvout
// CHECK: bb_mset

"memref.global"() {sym_name = "a_g", type = memref<32x16xi8>, initial_value = dense<1> : tensor<32x16xi8>, visibility = "private"} : () -> ()

func.func private @bb_test_report(i32) -> ()

func.func @grm_im2col_ref(%src : memref<32x16xi8>, %dst : memref<52x16xi8>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c13 = arith.constant 13 : index
  %c16 = arith.constant 16 : index
  %c52 = arith.constant 52 : index

  scf.for %i = %c0 to %c52 step %c1 {
    scf.for %j = %c0 to %c16 step %c1 {
      %z = arith.constant 0 : i8
      memref.store %z, %dst[%i, %j] : memref<52x16xi8>
    }
  }

  scf.for %r = %c0 to %c13 step %c1 {
    scf.for %c = %c0 to %c16 step %c1 {
      %w0 = arith.muli %r, %c16 : index
      %w = arith.addi %w0, %c : index
      scf.for %kr = %c0 to %c4 step %c1 {
        %lin0 = arith.muli %w, %c4 : index
        %lin = arith.addi %lin0, %kr : index
        %orow = arith.divui %lin, %c16 : index
        %ocol = arith.remui %lin, %c16 : index
        %sr = arith.addi %r, %kr : index
        %v = memref.load %src[%sr, %c] : memref<32x16xi8>
        memref.store %v, %dst[%orow, %ocol] : memref<52x16xi8>
      }
    }
  }
  return
}

func.func private @grm_cmp(%g : memref<52x16xi8>, %d : memref<52x16xi8>) -> i1 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %r = arith.constant 52 : index
  %c = arith.constant 16 : index
  %tr = arith.constant true
  %all = scf.for %i = %c0 to %r step %c1 iter_args(%ok_row = %tr) -> (i1) {
    %row_all = scf.for %j = %c0 to %c step %c1 iter_args(%ok_cell = %tr) -> (i1) {
      %vg = memref.load %g[%i, %j] : memref<52x16xi8>
      %vd = memref.load %d[%i, %j] : memref<52x16xi8>
      %eq = arith.cmpi eq, %vg, %vd : i8
      %both = arith.andi %ok_cell, %eq : i1
      scf.yield %both : i1
    }
    %both2 = arith.andi %ok_row, %row_all : i1
    scf.yield %both2 : i1
  }
  return %all : i1
}

func.func @main() -> i8 {
  %zero = arith.constant 0 : i8
  %one = arith.constant 1 : i8

  %c4 = arith.constant 4 : i64
  %c1 = arith.constant 1 : i64
  %c16 = arith.constant 16 : i64
  %c0 = arith.constant 0 : i64

  %a = memref.get_global @a_g : memref<32x16xi8>
  %b_grm = memref.alloc() : memref<52x16xi8>
  %b_dut = memref.alloc() : memref<52x16xi8>

  func.call @grm_im2col_ref(%a, %b_grm) : (memref<32x16xi8>, memref<52x16xi8>) -> ()

  "buckyball.bb_im2col"(%a, %b_dut, %c4, %c1, %c16, %c16, %c0, %c0)
      : (memref<32x16xi8>, memref<52x16xi8>, i64, i64, i64, i64, i64, i64) -> ()

  %match = func.call @grm_cmp(%b_grm, %b_dut) : (memref<52x16xi8>, memref<52x16xi8>) -> i1
  %true = arith.constant true
  %fail_i1 = arith.cmpi ne, %match, %true : i1
  %fail = arith.extui %fail_i1 : i1 to i32
  func.call @bb_test_report(%fail) : (i32) -> ()
  %rc = arith.select %fail_i1, %one, %zero : i8

  memref.dealloc %b_grm : memref<52x16xi8>
  memref.dealloc %b_dut : memref<52x16xi8>
  return %rc : i8
}
