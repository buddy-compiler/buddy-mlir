// RUN: buddy-opt %s -lower-linalg-to-vir -lower-vir-to-vector='vector-width=4' | FileCheck %s

#map0 = affine_map<(d0, d1) -> (d0, 0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (d1, d0)>

func.func @zero_broadcast_uses_transfer_read(%src: memref<2xf32>, %out: memref<2x2xf32>) {
  %expanded = memref.expand_shape %src [[0, 1]] output_shape [2, 1] : memref<2xf32> into memref<2x1xf32>
  linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%expanded : memref<2x1xf32>) outs(%out : memref<2x2xf32>) attrs = {broadcastDims = array<i64: 1>} {
  ^bb0(%in: f32, %out_elem: f32):
    linalg.yield %in : f32
  }
  return
}

func.func @transposed_output_uses_transfer_write(%src: memref<2x2xf32>, %out: memref<2x2xf32>) {
  linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["parallel", "parallel"]} ins(%src : memref<2x2xf32>) outs(%out : memref<2x2xf32>) {
  ^bb0(%in: f32, %out_elem: f32):
    linalg.yield %in : f32
  }
  return
}

// CHECK-LABEL: func.func @zero_broadcast_uses_transfer_read
// CHECK: %[[SV:.*]] = memref.subview
// CHECK: %[[READ:.*]] = vector.transfer_read %[[SV]]
// CHECK-NOT: vector.load %[[SV]]
// CHECK: vector.store %[[READ]],

// CHECK-LABEL: func.func @transposed_output_uses_transfer_write
// CHECK: %[[SRC:.*]] = memref.transpose %arg0
// CHECK: %[[DST:.*]] = memref.transpose %arg1
// CHECK: %[[V:.*]] = vector.load %[[SRC]]
// CHECK: vector.transfer_write %[[V]], %[[DST]]
// CHECK-NOT: vector.store %[[V]], %[[DST]]
