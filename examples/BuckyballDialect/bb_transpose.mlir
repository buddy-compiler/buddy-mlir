// RUN: buddy-opt %s \
// RUN:     -lower-buckyball | \
// RUN: FileCheck %s

// DUT: single `bb_transpose` (lowers to mset / mvin / bb_transpose / mvout / mset release).
// GRM: pure MLIR (`grm_transpose_ref`) — same as buckyball.c `transpose_u8_matrix` (L230–235):
//   dst[j*rows+i] = src[i*cols+j]. Full 1024×16 output compared element-wise.
// Host: debug_helper.c -> bb_test_report (only for TEST PASS line).
// CHECK: bb_mset
// CHECK: bb_mvin
// CHECK: bb_transpose
// CHECK: bb_mvout
// CHECK: bb_mset

"memref.global"() {sym_name = "a_g", type = memref<16x1024xi8>, initial_value = dense<7> : tensor<16x1024xi8>, visibility = "private"} : () -> ()

func.func private @bb_test_report(i32) -> ()

// -----------------------------------------------------------------------------
// GRM — golden reference (same loops as buckyball.c transpose_u8_matrix, L230–235)
// -----------------------------------------------------------------------------
func.func @grm_transpose_ref(%src : memref<16x1024xi8>, %dst : memref<1024x16xi8>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %rows = arith.constant 16 : index
  %cols = arith.constant 1024 : index
  scf.for %i = %c0 to %rows step %c1 {
    scf.for %j = %c0 to %cols step %c1 {
      %v = memref.load %src[%i, %j] : memref<16x1024xi8>
      memref.store %v, %dst[%j, %i] : memref<1024x16xi8>
    }
  }
  return
}

// -----------------------------------------------------------------------------
// GRM — compare full 1024×16 i8 tiles
// -----------------------------------------------------------------------------
func.func private @grm_cmp(%g : memref<1024x16xi8>, %d : memref<1024x16xi8>) -> i1 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %nr = arith.constant 1024 : index
  %nc = arith.constant 16 : index
  %tr = arith.constant true
  %all = scf.for %i = %c0 to %nr step %c1 iter_args(%ok_row = %tr) -> (i1) {
    %row_all = scf.for %j = %c0 to %nc step %c1 iter_args(%ok_cell = %tr) -> (i1) {
      %vg = memref.load %g[%i, %j] : memref<1024x16xi8>
      %vd = memref.load %d[%i, %j] : memref<1024x16xi8>
      %eq = arith.cmpi eq, %vg, %vd : i8
      %both = arith.andi %ok_cell, %eq : i1
      scf.yield %both : i1
    }
    %both2 = arith.andi %ok_row, %row_all : i1
    scf.yield %both2 : i1
  }
  return %all : i1
}

// -----------------------------------------------------------------------------
// DUT — Buckyball bb_transpose (accelerator path)
// -----------------------------------------------------------------------------

func.func @main() -> i8 {
  %zero = arith.constant 0 : i8
  %one = arith.constant 1 : i8

  %a = memref.get_global @a_g : memref<16x1024xi8>
  %b_grm = memref.alloc() : memref<1024x16xi8>
  %b_dut = memref.alloc() : memref<1024x16xi8>

  // --- GRM ---
  func.call @grm_transpose_ref(%a, %b_grm) : (memref<16x1024xi8>, memref<1024x16xi8>) -> ()

  // --- DUT ---
  "buckyball.bb_transpose"(%a, %b_dut) : (memref<16x1024xi8>, memref<1024x16xi8>) -> ()

  // --- compare GRM vs DUT ---
  %match = func.call @grm_cmp(%b_grm, %b_dut) : (memref<1024x16xi8>, memref<1024x16xi8>) -> i1
  %true = arith.constant true
  %fail_i1 = arith.cmpi ne, %match, %true : i1
  %fail = arith.extui %fail_i1 : i1 to i32
  func.call @bb_test_report(%fail) : (i32) -> ()
  %rc = arith.select %fail_i1, %one, %zero : i8

  memref.dealloc %b_grm : memref<1024x16xi8>
  memref.dealloc %b_dut : memref<1024x16xi8>
  return %rc : i8
}
