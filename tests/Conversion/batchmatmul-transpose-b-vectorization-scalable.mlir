// RUN: buddy-opt %s \
// RUN:     -batchmatmul-transpose-b-vectorization="vector-type=scalable vector-size=4" \
// RUN: | FileCheck %s

#map_a = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map_b = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#map_c = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>

func.func @batch_matmul_transpose_b_f32(%A: memref<4x8x16xf32>,
                                         %B: memref<4x32x16xf32>,
                                         %C: memref<4x8x32xf32>) {
  linalg.batch_matmul indexing_maps = [#map_a, #map_b, #map_c]
    ins(%A, %B: memref<4x8x16xf32>, memref<4x32x16xf32>)
    outs(%C: memref<4x8x32xf32>)
  return
}

// CHECK-LABEL: func.func @batch_matmul_transpose_b_f32
// CHECK:       vector.vscale
// CHECK:       arith.muli {{.*}} %vscale
// CHECK:       vector<[4]xf32>
// CHECK:       vector.load
// CHECK:       vector.fma
// CHECK:       vector.reduction
// CHECK:       return
// CHECK-NOT:   linalg.batch_matmul_transpose_b
