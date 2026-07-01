// RUN: buddy-opt %s -lower-buckyball | FileCheck %s
// Simple Quant test using Bank SSA operations
// CHECK: bb_mset
// CHECK: bb_mvin
// CHECK: bb_quant
// CHECK: bb_mvout
// CHECK: bb_mset

func.func private @bb_test_report(i32) -> ()

func.func @main() -> i8 {
  %zero = arith.constant 0 : i8
  %input = memref.alloc() : memref<16x16xf32>
  %output = memref.alloc() : memref<16x16xi8>

  // Initialize: input[i,j] = (i + j) * 0.5
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %n = arith.constant 16 : index
  %half = arith.constant 0.5 : f32
  scf.for %i = %c0 to %n step %c1 {
    scf.for %j = %c0 to %n step %c1 {
      %idx = arith.index_cast %i : index to i32
      %jdx = arith.index_cast %j : index to i32
      %sum = arith.addi %idx, %jdx : i32
      %sum_f = arith.sitofp %sum : i32 to f32
      %val = arith.mulf %sum_f, %half : f32
      memref.store %val, %input[%i, %j] : memref<16x16xf32>
    }
  }

  // Bank SSA operations
  %bank_in = "buckyball.bank_alloc"() : () -> i64
  %bank_out = "buckyball.bank_alloc"() : () -> i64
  %depth = arith.constant 16 : i64
  %stride = arith.constant 1 : i64
  %iter = arith.constant 256 : i64
  // 1.0f as i64
  %scale = arith.constant 1065353216 : i64

  %bank_in_loaded = "buckyball.bank_mvin"(%input, %bank_in, %depth, %stride)
    : (memref<16x16xf32>, i64, i64, i64) -> i64

  // Quant operation
  "buckyball.quant"(%bank_in_loaded, %bank_out, %iter, %scale)
    : (i64, i64, i64, i64) -> ()

  %bank_out_stored = "buckyball.bank_mvout"(%output, %bank_out, %depth, %stride)
    : (memref<16x16xi8>, i64, i64, i64) -> i64

  "buckyball.bank_release"(%bank_in_loaded) : (i64) -> ()
  "buckyball.bank_release"(%bank_out_stored) : (i64) -> ()

  %zero_i32 = arith.constant 0 : i32
  func.call @bb_test_report(%zero_i32) : (i32) -> ()

  memref.dealloc %input : memref<16x16xf32>
  memref.dealloc %output : memref<16x16xi8>
  return %zero : i8
}
