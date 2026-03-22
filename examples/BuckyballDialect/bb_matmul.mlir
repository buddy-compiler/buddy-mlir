// RUN: buddy-opt %s \
// RUN:     -lower-buckyball | \
// RUN: FileCheck %s

// DUT: single `bb_matmul` (lowers to mset / mvin / mul_warp16 / mvout / mset release).
// GRM: pure MLIR (`grm_matmul_ref`) — same math as buckyball.c `cpu_matmul` (L246–256): row-major
//   C[i,j] += A[i,k]*B[k,j]. Full 16×16 tile compared element-wise.
// Host: debug_helper.c -> bb_test_report (only for TEST PASS line).
// CHECK: bb_mset
// CHECK: bb_mvin
// CHECK: bb_mul_warp16
// CHECK: bb_mvout
// CHECK: bb_mset

"memref.global"() {sym_name = "a_g", type = memref<16x1024xi8>, initial_value = dense<1> : tensor<16x1024xi8>, visibility = "private"} : () -> ()
"memref.global"() {sym_name = "b_g", type = memref<1024x16xi8>, initial_value = dense<1> : tensor<1024x16xi8>, visibility = "private"} : () -> ()

func.func private @bb_test_report(i32) -> ()

// -----------------------------------------------------------------------------
// GRM — golden reference (same loops as buckyball.c cpu_matmul, L246–256)
// -----------------------------------------------------------------------------
func.func @grm_matmul_ref(%a : memref<16x1024xi8>, %b : memref<1024x16xi8>, %c : memref<16x16xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %r = arith.constant 16 : index
  %kmax = arith.constant 1024 : index
  scf.for %i = %c0 to %r step %c1 {
    scf.for %j = %c0 to %r step %c1 {
      %z = arith.constant 0 : i32
      %dot = scf.for %k = %c0 to %kmax step %c1 iter_args(%acc = %z) -> (i32) {
        %va = memref.load %a[%i, %k] : memref<16x1024xi8>
        %vb = memref.load %b[%k, %j] : memref<1024x16xi8>
        %vai = arith.extsi %va : i8 to i32
        %vbi = arith.extsi %vb : i8 to i32
        %p = arith.muli %vai, %vbi : i32
        %s = arith.addi %acc, %p : i32
        scf.yield %s : i32
      }
      memref.store %dot, %c[%i, %j] : memref<16x16xi32>
    }
  }
  return
}

// -----------------------------------------------------------------------------
// GRM — compare full 16×16 i32 tiles (any mismatch => false)
// -----------------------------------------------------------------------------
func.func private @grm_cmp(%g : memref<16x16xi32>, %d : memref<16x16xi32>) -> i1 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %n = arith.constant 16 : index
  %tr = arith.constant true
  %all = scf.for %i = %c0 to %n step %c1 iter_args(%ok_row = %tr) -> (i1) {
    %row_all = scf.for %j = %c0 to %n step %c1 iter_args(%ok_cell = %tr) -> (i1) {
      %vg = memref.load %g[%i, %j] : memref<16x16xi32>
      %vd = memref.load %d[%i, %j] : memref<16x16xi32>
      %eq = arith.cmpi eq, %vg, %vd : i32
      %both = arith.andi %ok_cell, %eq : i1
      scf.yield %both : i1
    }
    %both2 = arith.andi %ok_row, %row_all : i1
    scf.yield %both2 : i1
  }
  return %all : i1
}

// -----------------------------------------------------------------------------
// DUT — Buckyball bb_matmul (accelerator path)
// -----------------------------------------------------------------------------

func.func @main() -> i8 {
  %zero = arith.constant 0 : i8
  %one = arith.constant 1 : i8

  %a = memref.get_global @a_g : memref<16x1024xi8>
  %b = memref.get_global @b_g : memref<1024x16xi8>
  %c_grm = memref.alloc() : memref<16x16xi32>
  %c_dut = memref.alloc() : memref<16x16xi32>

  // --- GRM ---
  func.call @grm_matmul_ref(%a, %b, %c_grm) : (memref<16x1024xi8>, memref<1024x16xi8>, memref<16x16xi32>) -> ()

  // --- DUT ---
  "buckyball.bb_matmul"(%a, %b, %c_dut) :
    (memref<16x1024xi8>, memref<1024x16xi8>, memref<16x16xi32>) -> ()

  // --- compare GRM vs DUT (full tile) ---
  %match = func.call @grm_cmp(%c_grm, %c_dut) : (memref<16x16xi32>, memref<16x16xi32>) -> i1
  %true = arith.constant true
  %fail_i1 = arith.cmpi ne, %match, %true : i1
  %fail = arith.extui %fail_i1 : i1 to i32
  func.call @bb_test_report(%fail) : (i32) -> ()
  %rc = arith.select %fail_i1, %one, %zero : i8

  memref.dealloc %c_grm : memref<16x16xi32>
  memref.dealloc %c_dut : memref<16x16xi32>
  return %rc : i8
}
