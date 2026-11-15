// RUN: buddy-opt %s -split-input-file -lower-linalg-to-vir -verify-diagnostics \
// RUN: | FileCheck %s

// A linalg.reduce that lower-linalg-to-vir does not lower stays, for the
// other passes (loops), without a diagnostic.

// A rank-3 reduction of the last dim.
// CHECK-LABEL: func.func @reduce_3d_last_dim
// CHECK: linalg.reduce
// CHECK-NOT: vir.
func.func @reduce_3d_last_dim(%arg0: memref<2x4x8xf32>, %arg1: memref<2x4xf32>) {
  linalg.reduce ins(%arg0 : memref<2x4x8xf32>) outs(%arg1 : memref<2x4xf32>) dimensions = [2]
    (%in: f32, %init: f32) {
      %sum = arith.addf %in, %init : f32
      linalg.yield %sum : f32
    }
  return
}

// -----

// A bf16 sum: LLVM would compute it in another precision.
// CHECK-LABEL: func.func @reduce_bf16
// CHECK: linalg.reduce
// CHECK-NOT: vir.
func.func @reduce_bf16(%arg0: memref<16xbf16>, %arg1: memref<bf16>) {
  linalg.reduce ins(%arg0 : memref<16xbf16>) outs(%arg1 : memref<bf16>) dimensions = [0]
    (%in: bf16, %init: bf16) {
      %sum = arith.addf %in, %init : bf16
      linalg.yield %sum : bf16
    }
  return
}

// -----

// A combiner of two ops.
// CHECK-LABEL: func.func @reduce_two_ops
// CHECK: linalg.reduce
// CHECK-NOT: vir.
func.func @reduce_two_ops(%arg0: memref<16xf32>, %arg1: memref<f32>) {
  linalg.reduce ins(%arg0 : memref<16xf32>) outs(%arg1 : memref<f32>) dimensions = [0]
    (%in: f32, %init: f32) {
      %sq = arith.mulf %in, %in : f32
      %sum = arith.addf %sq, %init : f32
      linalg.yield %sum : f32
    }
  return
}

// -----

// A combiner of one argument only.
// CHECK-LABEL: func.func @reduce_not_of_arguments
// CHECK: linalg.reduce
// CHECK-NOT: vir.
func.func @reduce_not_of_arguments(%arg0: memref<16xi32>, %arg1: memref<i32>) {
  linalg.reduce ins(%arg0 : memref<16xi32>) outs(%arg1 : memref<i32>) dimensions = [0]
    (%in: i32, %init: i32) {
      %sum = arith.addi %in, %in : i32
      linalg.yield %sum : i32
    }
  return
}
