// RUN: buddy-opt %s -convert-tile-to-buckyball -lower-buckyball | FileCheck %s

// A[64,K] x B[K,N] -> C[64,N] with K=N=1024 (C is 64x1024). Tiling splits M/N; K stays one
// hardware tile along k (default bank_depth) so each bb_matmul sees full K for its (m,n) tile.
// GRM: pure MLIR. DUT: tile.tile_matmul. Host: debug_helper.c -> bb_test_report.
// CHECK: bb_mul_warp16

"memref.global"() {sym_name = "a_g", type = memref<64x1024xi8>, initial_value = dense<1> : tensor<64x1024xi8>, visibility = "private"} : () -> ()
"memref.global"() {sym_name = "b_g", type = memref<1024x1024xi8>, initial_value = dense<1> : tensor<1024x1024xi8>, visibility = "private"} : () -> ()

func.func private @bb_test_report(i32) -> ()

// -----------------------------------------------------------------------------
// GRM — reference matmul (same arithmetic as buckyball.c cpu_matmul)
// -----------------------------------------------------------------------------
func.func @grm_matmul_ref(%a : memref<64x1024xi8>, %b : memref<1024x1024xi8>, %c : memref<64x1024xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %m = arith.constant 64 : index
  %n = arith.constant 1024 : index
  %kdim = arith.constant 1024 : index
  scf.for %i = %c0 to %m step %c1 {
    scf.for %j = %c0 to %n step %c1 {
      %z = arith.constant 0 : i32
      %dot = scf.for %kk = %c0 to %kdim step %c1 iter_args(%acc = %z) -> (i32) {
        %va = memref.load %a[%i, %kk] : memref<64x1024xi8>
        %vb = memref.load %b[%kk, %j] : memref<1024x1024xi8>
        %vai = arith.extsi %va : i8 to i32
        %vbi = arith.extsi %vb : i8 to i32
        %p = arith.muli %vai, %vbi : i32
        %s = arith.addi %acc, %p : i32
        scf.yield %s : i32
      }
      memref.store %dot, %c[%i, %j] : memref<64x1024xi32>
    }
  }
  return
}

func.func @grm_cmp(%g : memref<64x1024xi32>, %d : memref<64x1024xi32>) -> i1 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %m = arith.constant 64 : index
  %n = arith.constant 1024 : index
  %tr = arith.constant true
  %all = scf.for %i = %c0 to %m step %c1 iter_args(%ok_row = %tr) -> (i1) {
    %row_all = scf.for %j = %c0 to %n step %c1 iter_args(%ok_cell = %tr) -> (i1) {
      %vg = memref.load %g[%i, %j] : memref<64x1024xi32>
      %vd = memref.load %d[%i, %j] : memref<64x1024xi32>
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
// DUT — tile.tile_matmul (lowers to bb_matmul subviews)
// -----------------------------------------------------------------------------

func.func @main() -> i8 {
  %zero = arith.constant 0 : i8
  %one = arith.constant 1 : i8

  %a = memref.get_global @a_g : memref<64x1024xi8>
  %b = memref.get_global @b_g : memref<1024x1024xi8>
  %c_grm = memref.alloc() : memref<64x1024xi32>
  %c_dut = memref.alloc() : memref<64x1024xi32>

  func.call @grm_matmul_ref(%a, %b, %c_grm) : (memref<64x1024xi8>, memref<1024x1024xi8>, memref<64x1024xi32>) -> ()

  "tile.tile_matmul"(%a, %b, %c_dut) : (memref<64x1024xi8>, memref<1024x1024xi8>, memref<64x1024xi32>) -> ()

  %match = func.call @grm_cmp(%c_grm, %c_dut) : (memref<64x1024xi32>, memref<64x1024xi32>) -> i1
  %true = arith.constant true
  %fail_i1 = arith.cmpi ne, %match, %true : i1
  %fail = arith.extui %fail_i1 : i1 to i32
  func.call @bb_test_report(%fail) : (i32) -> ()
  %rc = arith.select %fail_i1, %one, %zero : i8

  memref.dealloc %c_grm : memref<64x1024xi32>
  memref.dealloc %c_dut : memref<64x1024xi32>
  return %rc : i8
}
