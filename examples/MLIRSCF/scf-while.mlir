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
  %init = arith.constant 0 : i32
  %end = arith.constant 5 : i32
  %c2_i32 = arith.constant 2 : i32
  %res = scf.while (%arg0 = %init) : (i32) -> (i32) {
    // Before region.
    %cond = arith.cmpi slt, %arg0, %end : i32
    scf.condition(%cond) %arg0 : i32
  } do {
  // After region.
  ^bb0(%arg5: i32):
    %1 = arith.addi %arg5, %c2_i32 : i32
    scf.yield %1 : i32
  }
  // CHECK: 6
  vector.print %res : i32
  return
}
