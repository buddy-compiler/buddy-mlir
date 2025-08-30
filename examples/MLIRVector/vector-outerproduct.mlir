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
  %0 = arith.constant dense<[12, 11, 78, 90, 23, 56]> : vector<6xi32>
  // CHECK: ( 12, 11, 78, 90, 23, 56 )
  vector.print %0 : vector<6xi32>

  %1 = arith.constant dense<[13, 56, 78, 89, 95, 23]> : vector<6xi32>
  // CHECK: ( 13, 56, 78, 89, 95, 23 )
  vector.print %1 : vector<6xi32>

  %2 = vector.outerproduct %0, %1 : vector<6xi32>, vector<6xi32> // will give vector of shape 6x6
  // CHECK:    ( ( 156, 672, 936, 1068, 1140, 276 ),
  // CHECK-SAME: ( 143, 616, 858, 979, 1045, 253 ),
  // CHECK-SMAE: ( 1014, 4368, 6084, 6942, 7410, 1794 ),
  // CHECK-SMAE: ( 1170, 5040, 7020, 8010, 8550, 2070 ),
  // CHECK-SMAE: ( 299, 1288, 1794, 2047, 2185, 529 ),
  // CHECK-SMAE: ( 728, 3136, 4368, 4984, 5320, 1288 ) )
  vector.print %2 : vector<6x6xi32>

  %3 = arith.constant dense<[12, 67, 49]> : vector<3xi32>

  // CHECK: ( 12, 67, 49 )
  vector.print %3 : vector<3xi32>

  %4 = vector.outerproduct %0, %3 : vector<6xi32>, vector<3xi32> // will give vector of shape 6x3
  // CHECK:   ( ( 144, 804, 588 ),
  // CHECK-SAME: ( 132, 737, 539 ),
  // CHECK-SAME: ( 936, 5226, 3822 ),
  // CHECK-SAME: ( 1080, 6030, 4410 ),
  // CHECK-SMAE: ( 276, 1541, 1127 ),
  // CHECK-SMAE: ( 672, 3752, 2744 ) )
  vector.print %4 : vector<6x3xi32>

  %cons = arith.constant 4 : i32
  %5 = vector.outerproduct %0, %cons : vector<6xi32>, i32  // will give vector of same shape, formula is [a,c]*d = [a*d,c*d]
  // CHECK: ( 48, 44, 312, 360, 92, 224 )
  vector.print %5 : vector<6xi32>

  %ret = arith.constant 0 : i32
  return %ret : i32
}
