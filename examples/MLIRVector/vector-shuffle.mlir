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
  %0 = arith.constant dense<2.0> : vector<3x2xf32>
  // CHECK: ( ( 2, 2 ), ( 2, 2 ), ( 2, 2 ) )
  vector.print %0 : vector<3x2xf32>

  %1 = arith.constant dense<3.0> : vector <3x2xf32>
  // CHECK: ( ( 3, 3 ), ( 3, 3 ), ( 3, 3 ) )
  vector.print %1 : vector<3x2xf32>

  %2 = vector.shuffle %0, %1[1, 0, 3, 2] : vector<3x2xf32>, vector<3x2xf32>
  // CHECK: ( ( 2, 2 ), ( 2, 2 ), ( 3, 3 ), ( 2, 2 ) )
  vector.print %2 : vector<4x2xf32>

  %3 = arith.constant dense<4.0> : vector<4x3xf32>
  // CHECK: ( ( 4, 4, 4 ), ( 4, 4, 4 ), ( 4, 4, 4 ), ( 4, 4, 4 ) )
  vector.print %3 : vector<4x3xf32>

  %4 = arith.constant dense<5.0> : vector<5x3xf32>
  // CHECK: ( ( 5, 5, 5 ), ( 5, 5, 5 ), ( 5, 5, 5 ), ( 5, 5, 5 ), ( 5, 5, 5 ) )
  vector.print %4 : vector<5x3xf32>

  %5 = vector.shuffle %3, %4[1, 7, 5, 3, 4, 2] : vector<4x3xf32>, vector<5x3xf32>
  // CHECK: ( ( 4, 4, 4 ), ( 5, 5, 5 ), ( 5, 5, 5 ), ( 4, 4, 4 ), ( 5, 5, 5 ), ( 4, 4, 4 ) )
  vector.print %5 : vector<6x3xf32>

  %6 = arith.constant dense<4.0> : vector<2x2xf32>
  // CHECK: ( ( 4, 4 ), ( 4, 4 ) )
  vector.print %6 : vector<2x2xf32>

  %7 = arith.constant dense<10.0> : vector<2x2xf32>
  // CHECK: ( ( 10, 10 ), ( 10, 10 ) )
  vector.print %7 : vector<2x2xf32>

  %8 = vector.shuffle %6, %7[0, 1, 2, 3] : vector<2x2xf32>, vector<2x2xf32>
  // CHECK: ( ( 4, 4 ), ( 4, 4 ), ( 10, 10 ), ( 10, 10 ) )
  vector.print %8 : vector<4x2xf32>

  %ret = arith.constant 0 : i32
  return %ret : i32
}
