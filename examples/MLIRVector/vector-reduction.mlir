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
  %0 = arith.constant dense<[12, 13, 14, 15, 16, 90]> : vector<6xi32>
  // CHECK: ( 12, 13, 14, 15, 16, 90 )
  vector.print %0 : vector<6xi32>

  %sum = vector.reduction <add>, %0 : vector<6xi32> into i32
  // CHECK: 160
  vector.print %sum : i32

  %mul = vector.reduction <mul>, %0 : vector<6xi32> into i32
  // CHECK: 47174400
  vector.print %mul : i32

  %xor = vector.reduction <xor>, %0 : vector<6xi32> into i32
  // CHECK: 74
  vector.print %xor : i32

  %and = vector.reduction <and>, %0 : vector<6xi32> into i32
  // CHECK: 0
  vector.print %and : i32

  %or = vector.reduction <or>, %0 : vector<6xi32> into i32
  // CHECK: 95
  vector.print %or : i32

  %ret = arith.constant 0 : i32
  return %ret : i32
}
