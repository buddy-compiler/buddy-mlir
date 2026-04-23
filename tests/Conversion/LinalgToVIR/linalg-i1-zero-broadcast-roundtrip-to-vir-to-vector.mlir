// RUN: buddy-opt %s -lower-linalg-to-vir -lower-vir-to-vector="vector-width=4" -cse --convert-vector-to-scf --expand-strided-metadata --lower-affine --convert-scf-to-cf --convert-cf-to-llvm --convert-vector-to-llvm --finalize-memref-to-llvm --convert-arith-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts | mlir-runner -O0 -e main -entry-point-result=i32 -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext,%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext | FileCheck %s

// Regression test for mask values that are materialized as i1 memrefs,
// broadcast through a zero-stride view, and later consumed by scalar code.
// This matches the pattern behind Triton's gather_column_ld_mask failure.

#broadcast_row = affine_map<(d0, d1) -> (d0, 0)>
#id = affine_map<(d0, d1) -> (d0, d1)>

module {
  func.func @main() -> i32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32

    %src = memref.alloc() : memref<2xi32>
    %mask1 = memref.alloc() : memref<2x1xi1>
    %mask2 = memref.alloc() : memref<2x4xi1>

    affine.for %i = 0 to 2 {
      %iv = arith.index_cast %i : index to i32
      memref.store %iv, %src[%i] : memref<2xi32>
    }

    %expanded = memref.expand_shape %src [[0, 1]] output_shape [2, 1]
      : memref<2xi32> into memref<2x1xi32>
    linalg.generic
        {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]}
        ins(%expanded : memref<2x1xi32>)
        outs(%mask1 : memref<2x1xi1>) {
      ^bb0(%a: i32, %m: i1):
        %pred = arith.cmpi slt, %a, %one : i32
        linalg.yield %pred : i1
    }

    linalg.generic
        {indexing_maps = [#broadcast_row, #id],
         iterator_types = ["parallel", "parallel"]}
        ins(%mask1 : memref<2x1xi1>)
        outs(%mask2 : memref<2x4xi1>) attrs = {broadcastDims = array<i64: 1>} {
      ^bb0(%m_in: i1, %m_out: i1):
        linalg.yield %m_in : i1
    }

    %sum = scf.for %i = %c0 to %c2 step %c1 iter_args(%acc0 = %zero) -> (i32) {
      %row = scf.for %j = %c0 to %c4 step %c1 iter_args(%acc1 = %acc0) -> (i32) {
        %m = memref.load %mask2[%i, %j] : memref<2x4xi1>
        %inc = arith.select %m, %one, %zero : i32
        %next = arith.addi %acc1, %inc : i32
        scf.yield %next : i32
      }
      scf.yield %row : i32
    }

    return %sum : i32
  }
}

// CHECK: 4
