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
  // vector.shape_cast cast vector<... x a1 x a2 x ... x an x ... x T> to
  // vector<... x b x ... x T>, where a1 * a2 * ... * an == b

  %0 = arith.constant dense<[
    0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18.,
    19., 20., 21., 22., 23., 24., 25., 26., 27., 28., 29., 30., 31., 32., 33., 34., 35.
  ]> : vector<36xf32>

  // This cast should not move any element.
  %1 = vector.shape_cast %0 : vector<36xf32> to vector<3x2x2x3xf32>
  // CHECK:  ( ( ( ( 0, 1, 2 ),   ( 3, 4, 5 ) ),
  // CHECK-SAME: ( ( 6, 7, 8 ), ( 9, 10, 11 ) ) ),
  // CHECK-SAME: ( ( ( 12, 13, 14 ), ( 15, 16, 17 ) ),
  // CHECK-SAME: ( ( 18, 19, 20 ), ( 21, 22, 23 ) ) ),
  // CHECK-SAME: ( ( ( 24, 25, 26 ), ( 27, 28, 29 ) ),
  // CHECK-SAME: ( ( 30, 31, 32 ), ( 33, 34, 35 ) ) ) )
  vector.print %1 : vector<3x2x2x3xf32>

  %2 = vector.shape_cast %1 : vector<3x2x2x3xf32> to vector<3x4x3xf32>
  // CHECK:   ( ( ( 0, 1, 2 ), ( 3, 4, 5 ), ( 6, 7, 8 ),
  // CHECK-SAME: ( 9, 10, 11 ) ), ( ( 12, 13, 14 ), ( 15, 16, 17 ),
  // CHECK-SAME: ( 18, 19, 20 ), ( 21, 22, 23 ) ), ( ( 24, 25, 26 ),
  // CHECK-SAME: ( 27, 28, 29 ), ( 30, 31, 32 ), ( 33, 34, 35 ) ) )
  vector.print %2 : vector<3x4x3xf32>

  %ret = arith.constant 0 : i32
  return %ret : i32
}
