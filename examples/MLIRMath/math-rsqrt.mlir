// RUN: buddy-opt %s \
// RUN:     -convert-vector-to-scf -convert-scf-to-cf -convert-cf-to-llvm \
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
  %v0 = arith.constant dense<[1., 2., 3., 4., 5., 6., 7., 8.]> : vector<8xf32>
  %0 = math.rsqrt %v0 : vector<8xf32>
  // Since floating point precision may differ across various hardware platforms,
  // we are only verifying results to three decimal places of precision here.
  // CHECK: {{( 1, 0.707[0-9]*, 0.577[0-9]*, 0.5, 0.447[0-9]*, 0.408[0-9]*, 0.377[0-9]*, 0.353[0-9]* )}}
  vector.print %0 : vector<8xf32>
  func.return
}
