// RUN: buddy-opt %s -lower-linalg-to-vir | FileCheck %s

#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0, 0)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>

func.func @index_and_zero_broadcast_regression(%src: memref<2xf32>, %out: memref<2x2xf32>) {
  %zero = arith.constant 0.000000e+00 : f32
  %tmp_idx = memref.alloc() : memref<2xi32>
  linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%tmp_idx : memref<2xi32>) {
  ^bb0(%out_elem: i32):
    %idx = linalg.index 0 : index
    %v = arith.index_cast %idx : index to i32
    linalg.yield %v : i32
  }
  linalg.fill ins(%zero : f32) outs(%out : memref<2x2xf32>)
  %expanded = memref.expand_shape %src [[0, 1]] output_shape [2, 1] : memref<2xf32> into memref<2x1xf32>
  linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["parallel", "parallel"]} ins(%expanded : memref<2x1xf32>) outs(%out : memref<2x2xf32>) attrs = {broadcastDims = array<i64: 1>} {
  ^bb0(%in: f32, %out_elem: f32):
    linalg.yield %in : f32
  }
  return
}

// CHECK-LABEL: func.func @index_and_zero_broadcast_regression
// CHECK-NOT: linalg.index
// CHECK-NOT: linalg.generic
// CHECK: scf.for
// CHECK: arith.index_cast
// CHECK: memref.store
// CHECK: %[[EXP:.*]] = memref.expand_shape %arg0
// CHECK: vir.set_vl %[[VL:.*]] : index {
// CHECK:   %[[TEXP:.*]] = memref.transpose %[[EXP]] (d0, d1) -> (d0, d1) : memref<2x1xf32> to memref<2x1xf32, strided<[1, 1]>>
// CHECK:   %[[SV:.*]] = memref.subview %[[TEXP]][0, 0] [2, 2] [1, 0] : memref<2x1xf32, strided<[1, 1]>> to memref<2x2xf32, strided<[1, 0]>>
// CHECK:   %[[TOUT:.*]] = memref.transpose %{{.*}} (d0, d1) -> (d0, d1) : memref<2x2xf32> to memref<2x2xf32, strided<[2, 1]>>
// CHECK:   %[[V:.*]] = vir.load %[[SV]][] : memref<2x2xf32, strided<[1, 0]>> -> !vir.vec<2x?xf32>
// CHECK-NOT: !vir.vec<2x2xf32>
// CHECK:   vir.store %[[V]], %[[TOUT]][]
