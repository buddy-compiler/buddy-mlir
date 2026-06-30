// RUN: buddy-opt %s -convert-tile-to-buckyball -lower-buckyball | FileCheck %s
// Tile-level matrix multiplication test using Tile dialect
// This example uses tile.tile_matmul which is lowered to Buckyball operations
// CHECK: bb_mset
// CHECK: bb_mvin
// CHECK: bb_mul_warp16
// CHECK: bb_mvout
// CHECK: bb_mset

func.func private @bb_test_report(i32) -> ()

func.func @main() -> i8 {
  %zero = arith.constant 0 : i8
  %a = memref.alloc() : memref<32x32xi8>
  %b = memref.alloc() : memref<32x32xi8>
  %c = memref.alloc() : memref<32x32xi32>

  // Initialize: a[i,j] = 1, b[i,j] = 1
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %n = arith.constant 32 : index
  %one_i8 = arith.constant 1 : i8
  %zero_i32 = arith.constant 0 : i32

  scf.for %i = %c0 to %n step %c1 {
    scf.for %j = %c0 to %n step %c1 {
      memref.store %one_i8, %a[%i, %j] : memref<32x32xi8>
      memref.store %one_i8, %b[%i, %j] : memref<32x32xi8>
      memref.store %zero_i32, %c[%i, %j] : memref<32x32xi32>
    }
  }

  // Tile-level matrix multiplication using Tile dialect
  // This will be lowered to Buckyball mul_warp16 operations with automatic tiling
  tile.tile_matmul %a %b %c : memref<32x32xi8> memref<32x32xi8> memref<32x32xi32>

  // Verify: c[0,0] should be 32 (sum of 32 multiplications of 1*1)
  %expected = arith.constant 32 : i32
  %result = memref.load %c[%c0, %c0] : memref<32x32xi32>
  %fail_i1 = arith.cmpi ne, %result, %expected : i32
  %fail = arith.extui %fail_i1 : i1 to i32
  func.call @bb_test_report(%fail) : (i32) -> ()

  memref.dealloc %a : memref<32x32xi8>
  memref.dealloc %b : memref<32x32xi8>
  memref.dealloc %c : memref<32x32xi32>
  return %zero : i8
}
