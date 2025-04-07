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

#map0 = affine_map<(d0, d1) -> (d1, d0)>
memref.global "private" @gv : memref<4x4xf32> = dense<[[0. , 1. , 2. , 3. ],
                                                       [10., 11., 12., 13.],
                                                       [20., 21., 22., 23.],
                                                       [30., 31., 32., 33.]]>

func.func @main() -> (i32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %f0 = arith.constant 0. : f32
  %f1 = arith.constant 1. : f32
  %cst0 = arith.constant 0 : i32
  %mem = memref.get_global @gv : memref<4x4xf32>
  %v0 = vector.transfer_read %mem[%c0, %c0], %f0 { permutation_map = #map0 } : memref<4x4xf32>, vector<4x4xf32>
  %v1 = vector.transfer_read %mem[%c1, %c1], %f0 { permutation_map = #map0 } : memref<4x4xf32>, vector<2x3xf32>
  %v2 = vector.transfer_read %mem[%c0, %c0], %f0 {} : memref<4x4xf32>, vector<5x5xf32>
  %v3 = vector.transfer_read %mem[%c0, %c0], %f1 {} : memref<4x4xf32>, vector<5x5xf32>
  // CHECK: ( ( 0, 10, 20, 30 ), ( 1, 11, 21, 31 ), ( 2, 12, 22, 32 ), ( 3, 13, 23, 33 ) )
  vector.print %v0 : vector<4x4xf32>
  // CHECK: ( ( 11, 21, 31 ), ( 12, 22, 32 ) )
  vector.print %v1 : vector<2x3xf32>
  // CHECK: ( ( 0, 1, 2, 3, 0 ),
  // CHECK-SAME: ( 10, 11, 12, 13, 0 ),
  // CHECK-SAME: ( 20, 21, 22, 23, 0 ),
  // CHECK-SAME: ( 30, 31, 32, 33, 0 ),
  // CHECK-SMAE: ( 0, 0, 0, 0, 0 ) )
  vector.print %v2 : vector<5x5xf32>
  // CHECK: ( ( 0, 1, 2, 3, 1 ),
  // CHECK-SAME: ( 10, 11, 12, 13, 1 ),
  // CHECK-SMAE: ( 20, 21, 22, 23, 1 ),
  // CHECK-SMAE: ( 30, 31, 32, 33, 1 ),
  // CHECK-SMAE: ( 1, 1, 1, 1, 1 ) )
  vector.print %v3 : vector<5x5xf32>
  return  %cst0 : i32
}
