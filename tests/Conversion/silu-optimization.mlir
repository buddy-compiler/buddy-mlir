// RUN: buddy-opt -silu-optimization="vector-size=8" %s | FileCheck %s

// CHECK: #map = affine_map<(d0) -> (d0)>
// CHECK: module {
// CHECK:   func.func @silu_tosa(%arg0: memref<1x40x8960xf32, strided<[?, ?, ?], offset: ?>>) -> memref<1x40x8960xf32> {
// CHECK:%cst = arith.constant 1.000000e+00 : f32
// CHECK:     %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x40x8960xf32>
// CHECK:     %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<1x40x8960xf32>
// CHECK:     %c0 = arith.constant 0 : index
// CHECK:     %cst_1 = arith.constant 1.000000e+00 : f32
// CHECK:     %0 = vector.broadcast %cst_1 : f32 to vector<8xf32>
// CHECK:     %cst_2 = arith.constant 0.000000e+00 : f32
// CHECK:     %c0_3 = arith.constant 0 : index
// CHECK:     %dim = memref.dim %arg0, %c0_3 : memref<1x40x8960xf32, strided<[?, ?, ?], offset: ?>>
// CHECK:     %c1 = arith.constant 1 : index
// CHECK:     %dim_4 = memref.dim %arg0, %c1 : memref<1x40x8960xf32, strided<[?, ?, ?], offset: ?>>
// CHECK:     %c2 = arith.constant 2 : index
// CHECK:     %dim_5 = memref.dim %arg0, %c2 : memref<1x40x8960xf32, strided<[?, ?, ?], offset: ?>>
// CHECK:     affine.for %arg1 = #map(%c0) to #map(%dim) {
// CHECK:       affine.for %arg2 = #map(%c0) to #map(%dim_4) {
// CHECK:         affine.for %arg3 = #map(%c0) to #map(%dim_5) step 8 {
// CHECK:           %1 = vector.transfer_read %arg0[%arg1, %arg2, %arg3], %cst_2 : memref<1x40x8960xf32, strided<[?, ?, ?], offset: ?>>, vector<8xf32>
// CHECK:           %2 = arith.negf %1 : vector<8xf32>
// CHECK:           %3 = math.exp %2 : vector<8xf32>
// CHECK:           %4 = arith.addf %0, %3 : vector<8xf32>
// CHECK:           %5 = arith.divf %0, %4 : vector<8xf32>
// CHECK:           %6 = arith.mulf %1, %5 : vector<8xf32>
// CHECK:           vector.transfer_write %6, %alloc_0[%arg1, %arg2, %arg3] : vector<8xf32>, memref<1x40x8960xf32>
// CHECK:         }
// CHECK:       }
// CHECK:     }
// CHECK:     return %alloc_0 : memref<1x40x8960xf32>
// CHECK:   }
// CHECK: }

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
func.func @silu_tosa(%arg0: memref<1x40x8960xf32, strided<[?, ?, ?], offset: ?>>) -> memref<1x40x8960xf32> {
  %cst = arith.constant 1.000000e+00 : f32
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x40x8960xf32>
  linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0 : memref<1x40x8960xf32, strided<[?, ?, ?], offset: ?>>) outs(%alloc : memref<1x40x8960xf32>) {
  ^bb0(%in: f32, %out: f32):
    %3 = arith.negf %in : f32
    %4 = math.exp %3 : f32
    %5 = arith.addf %4, %cst : f32
    %6 = arith.divf %cst, %5 : f32
    linalg.yield %6 : f32
  }
  %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<1x40x8960xf32>
  linalg.generic {indexing_maps = [#map, #map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0, %alloc : memref<1x40x8960xf32, strided<[?, ?, ?], offset: ?>>, memref<1x40x8960xf32>) outs(%alloc_0 : memref<1x40x8960xf32>) {
  ^bb0(%in: f32, %in_1: f32, %out: f32):
    %3 = arith.mulf %in, %in_1 : f32
    linalg.yield %3 : f32
  }
  return %alloc_0 : memref<1x40x8960xf32>
}
