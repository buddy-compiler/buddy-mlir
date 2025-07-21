// RUN: buddy-opt %s \
// RUN:     -lower-affine -convert-scf-to-cf -convert-cf-to-llvm -convert-vector-to-llvm \
// RUN:		  -finalize-memref-to-llvm -convert-func-to-llvm -convert-arith-to-llvm \
// RUN:		  -reconcile-unrealized-casts \
// RUN: | mlir-runner -e main -entry-point-result=void \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN: | FileCheck %s

func.func @main() {
  %c0 = arith.constant 1 : index
  %max0 = affine.max affine_map<(d0) -> (0, d0)> (%c0)
  // CHECK: 1
  vector.print %max0 : index
  %min = affine.min affine_map<(d0) -> (0, d0)> (%c0)
  // CHECK: 0
  vector.print %min : index
  func.return
}
