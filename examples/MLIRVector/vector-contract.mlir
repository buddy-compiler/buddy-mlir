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

#map0 = affine_map<(i, j, k) -> (i, j)>
#map1 = affine_map<(i, j, k) -> (j, k)>
#map2 = affine_map<(i, j, k) -> (i, k)>

func.func @main() -> (i32) {
  %c0 = arith.constant 0 : i32
  %v0 = arith.constant dense<[[1., 2., 3., 4.],
                              [5., 6., 7., 8.],
                              [9., 10., 11., 12.]]> : vector<3x4xf32>

  %v1 = arith.constant dense<[[1., 2., 3.],
                              [4., 5., 6.],
                              [7., 8., 9.],
                              [10., 11., 12.]]> : vector<4x3xf32>

  %v2 = arith.constant dense<[[0., 0., 0.],
                              [0., 0., 0.],
                              [0., 0., 0.]]> : vector<3x3xf32>

  %v3 = vector.contract {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "reduction", "parallel"]}
  %v0, %v1, %v2 : vector<3x4xf32>, vector<4x3xf32> into vector<3x3xf32>
  // CHECK: ( ( 70, 80, 90 ), ( 158, 184, 210 ), ( 246, 288, 330 ) )
  vector.print %v3 : vector<3x3xf32>
  return %c0 : i32
}
