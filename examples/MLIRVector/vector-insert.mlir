// RUN: buddy-opt %s \
// RUN:     -convert-vector-to-scf -convert-scf-to-cf \
// RUN:     -convert-cf-to-llvm \
// RUN:     -convert-vector-to-llvm -finalize-memref-to-llvm -convert-func-to-llvm \
// RUN:     -convert-arith-to-llvm \
// RUN:     -split-input-file -verify-diagnostics \
// RUN:     -reconcile-unrealized-casts \
// RUN: | mlir-runner -e main -entry-point-result=i32 \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN: | FileCheck %s

func.func @main() -> i32 {
  // vector.insert can insert scalar/sub-vector into a vector, creating a new
  // vector as result. The original vector will NOT change.

  // vector.insert only support literal as indices, if you need to insert
  // something into a vector with runtime values as indices, you need to cast
  // your base vector to 1-D vector and use vector.insertelement instead.

  %base = arith.constant dense<[[0, 1, 2], [10, 11, 12], [20, 21, 22]]>
    : vector<3x3xi32>


  // Insert a scalar into a vector:
  // It will insert scalar 100 into position [0, 0] at %base,
  // replacing original value 0.
  %c0 = arith.constant 100 : i32
  %v0 = vector.insert %c0, %base[0, 0] : i32 into vector<3x3xi32>
  // CHECK: ( ( 100, 1, 2 ), ( 10, 11, 12 ), ( 20, 21, 22 ) )
  vector.print %v0 : vector<3x3xi32>


  // Insert a sub-vector into a vector:
  // It will insert sub-vector %w1 into position [0, 1] at %base,
  // replacing original vector [10, 11, 12].
  %w1 = arith.constant dense<[100, 101, 102]> : vector<3xi32>
  %v1 = vector.insert %w1, %base[1] : vector<3xi32> into vector<3x3xi32>
  // CHECK: ( ( 0, 1, 2 ), ( 100, 101, 102 ), ( 20, 21, 22 ) )
  vector.print %v1 : vector<3x3xi32>


  // For edge case, you can even "insert" a vector with exactly same rank with
  // original vector. In this case, the result will be just the new vector itself.
  %w2 = arith.constant dense<[[200, 201, 202], [210, 211, 212], [220, 221, 222]]> : vector<3x3xi32>
  %v2 = vector.insert %w2, %base[] : vector<3x3xi32> into vector<3x3xi32>
  // CHECK: ( ( 200, 201, 202 ), ( 210, 211, 212 ), ( 220, 221, 222 ) )
  vector.print %v2 : vector<3x3xi32> // will print %w2


  // After all the insertions, the original vector does NOT change.
  // vector.insert creates new vector for result. It will never change the old vector
  // CHECK: ( ( 0, 1, 2 ), ( 10, 11, 12 ), ( 20, 21, 22 ) )
  vector.print %base : vector<3x3xi32>

  %ret = arith.constant 0 : i32
  return %ret : i32
}
