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
  %con = arith.constant 1 : i1 // If %con equals 0, the result will be 0.0.
  %lb = arith.constant 0 : index
  %ub = arith.constant 5 : index
  %initial = arith.constant 25.0 : f32
  %step = arith.constant 1 : index
  %t = arith.constant 5.0 : f32
  %final = scf.if %con -> f32 {
    %res = scf.for %iv = %lb to %ub step %step
    iter_args(%resiter = %initial) -> f32 {

      %1 = arith.addf %resiter , %t : f32
      scf.yield %1 : f32
    }
    scf.yield %res : f32
  } else {
    %res = scf.for %iv = %lb to %ub step %step
    iter_args(%resiter = %initial) -> f32 {
      %1 = arith.subf %resiter , %t : f32
      scf.yield %1 : f32
    }
    scf.yield %res : f32
  }
  // CHECK: 50
  vector.print %final : f32
  return
}
