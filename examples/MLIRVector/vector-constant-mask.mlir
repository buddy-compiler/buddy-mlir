// RUN: buddy-opt %s \
// RUN:     -convert-vector-to-scf -convert-scf-to-cf \
// RUN:     -convert-cf-to-llvm \
// RUN:     -convert-vector-to-llvm -finalize-memref-to-llvm -convert-func-to-llvm \
// RUN:     -convert-arith-to-llvm \
// RUN:     -reconcile-unrealized-casts \
// RUN: | mlir-runner -e main -entry-point-result=i32 \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN: | FileCheck %s

func.func @main() -> i32 {
  // constant_mask is the constant version of create_mask, with additional bound
  // check. It accepts a list of bounds for each dimension, to create a
  // hyper-rectangular region with 1s, and the rest of 0s.

  // This will create a [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  %mask1 = vector.constant_mask [5] : vector<10xi1>
  // CHECK: ( 1, 1, 1, 1, 1, 0, 0, 0, 0, 0 )
  vector.print %mask1 : vector<10xi1>

  // This will create a 2x2x2 region of 1s, and the rest for 0s.
  %mask2 = vector.constant_mask [2, 2, 2] : vector<3x3x3xi1>
  // CHECK: ( ( ( 1, 1, 0 ), ( 1, 1, 0 ), ( 0, 0, 0 ) ),
  // CHECK-SAME: ( ( 1, 1, 0 ), ( 1, 1, 0 ), ( 0, 0, 0 ) ),
  // CHECK-SAME: ( ( 0, 0, 0 ), ( 0, 0, 0 ), ( 0, 0, 0 ) ) )
  vector.print %mask2 : vector<3x3x3xi1>

  // It will perform a bound check, so the operation below is not allowed
  // %mask3 = vector.constant_mask [3, 3] : vector<2x2xi1>

  %ret = arith.constant 0 : i32
  return %ret : i32
}
