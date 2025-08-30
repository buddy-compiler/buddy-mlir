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
  // vector.splat fills a value into all elements of a vector, like "std::fill"
  // in C++.
  %0 = arith.constant 10.0 : f32

  // Creates a vector of shape 3x2, filling %0 as all elements
  %1 = vector.splat %0 : vector<3x2xf32>
  // CHECK: ( ( 10, 10 ), ( 10, 10 ), ( 10, 10 ) )
  vector.print %1 : vector<3x2xf32>

  // Doing the same thing with arith.constant
  %2 = arith.constant dense<10.0> : vector<3x2xf32>
  // CHECK: ( ( 10, 10 ), ( 10, 10 ), ( 10, 10 ) )
  vector.print %2 : vector<3x2xf32>

  // vector.splat can accept runtime values
  // while arith.constant only accepts constants.

  %ret = arith.constant 0 : i32
  return %ret : i32
}
