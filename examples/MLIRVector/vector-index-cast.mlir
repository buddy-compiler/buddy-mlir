// RUN: buddy-opt %s \
// RUN:     -convert-vector-to-scf -lower-affine -convert-scf-to-cf \
// RUN:     -convert-cf-to-llvm \
// RUN:     -convert-vector-to-llvm -finalize-memref-to-llvm -convert-func-to-llvm \
// RUN:     -convert-arith-to-llvm \
// RUN:     -reconcile-unrealized-casts \
// RUN: | mlir-runner -e main -entry-point-result=i32 \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN: | FileCheck %s

func.func @main() -> i32 {

  %v0 = arith.constant dense<[0, 0, 1, 1]> : vector<4xindex>
  // CHECK: ( 0, 0, 1, 1 )
  vector.print %v0 : vector<4xindex>

  %v1 = arith.index_cast %v0 : vector<4xindex> to vector<4xi1>
  // CHECK: ( 0, 0, 1, 1 )
  vector.print %v1 : vector<4xi1>

  %ret = arith.constant 0 : i32
  return %ret : i32
}
