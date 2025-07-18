// RUN: buddy-opt %s \
// RUN:     -convert-scf-to-cf \
// RUN:     -convert-vector-to-llvm \
// RUN:     -convert-func-to-llvm \
// RUN:     -reconcile-unrealized-casts \
// RUN: | mlir-runner -e main -entry-point-result=void \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN: | FileCheck %s

func.func @main() {
  %lb = arith.constant 0 : index
  %ub = arith.constant 5 : index
  %step = arith.constant 1 : index
  %sum_0 = arith.constant 0.0 : f32
  %t = arith.constant 5.0 : f32
  %sum = scf.for %iv = %lb to %ub step %step
  iter_args(%sum_iter = %sum_0) -> (f32) {
    %1 = arith.addf %sum_iter , %t : f32
    scf.yield %1 : f32
  }
  // CHECK: 25
  vector.print %sum : f32
  return
}
