// RUN: buddy-opt %s \
// RUN:     -convert-vector-to-llvm \
// RUN:     -convert-math-to-llvm \
// RUN:     -convert-func-to-llvm \
// RUN:     -convert-arith-to-llvm \
// RUN:     -reconcile-unrealized-casts \
// RUN: | mlir-runner -e main -entry-point-result=void \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN: | FileCheck %s

func.func @main() {
  %c1 = arith.constant 1.0 : f32
  %res = math.atan2 %c1, %c1 : f32
  // Since floating point precision may differ across various hardware platforms,
  // we are only verifying results to three decimal places of precision here.
  // CHECK: {{0.785[0-9]+}}
  vector.print %res : f32
  func.return
}
