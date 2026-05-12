// RUN: buddy-opt %s -split-input-file -lower-linalg-to-vir -lower-vir-to-vector='vector-width=4' -cse | FileCheck %s

#map1 = affine_map<(d0) -> (d0)>

func.func @store_only_scatter_i32_offsets(%offsets: memref<8xi32>,
                                          %values: memref<8xf32>,
                                          %out: memref<?xf32>) {
  linalg.generic {
      indexing_maps = [#map1, #map1],
      iterator_types = ["parallel"]
    } ins(%offsets, %values : memref<8xi32>, memref<8xf32>) {
  ^bb0(%idx_i32: i32, %value: f32):
    %idx = arith.index_cast %idx_i32 : i32 to index
    memref.store %value, %out[%idx] : memref<?xf32>
    linalg.yield
  }
  return
}

// CHECK-LABEL: func.func @store_only_scatter_i32_offsets
// CHECK-NOT: linalg.generic
// CHECK: vector.scatter
// CHECK-NOT: linalg.generic

// -----

#map1 = affine_map<(d0) -> (d0)>

func.func @store_only_masked_scatter_i32_offsets(%offsets: memref<8xi32>,
                                                 %values: memref<8xi32>,
                                                 %mask: memref<8xi1>,
                                                 %out: memref<?xi32>) {
  linalg.generic {
      indexing_maps = [#map1, #map1, #map1],
      iterator_types = ["parallel"]
    } ins(%offsets, %values, %mask : memref<8xi32>, memref<8xi32>, memref<8xi1>) {
  ^bb0(%idx_i32: i32, %value: i32, %mask_value: i1):
    %idx = arith.index_cast %idx_i32 : i32 to index
    %old = memref.load %out[%idx] : memref<?xi32>
    %selected = arith.select %mask_value, %value, %old : i32
    memref.store %selected, %out[%idx] : memref<?xi32>
    linalg.yield
  }
  return
}

// CHECK-LABEL: func.func @store_only_masked_scatter_i32_offsets
// CHECK-NOT: linalg.generic
// CHECK: vector.scatter
// CHECK-SAME: vector<4xi1>
// CHECK-NOT: linalg.generic

// -----

#map2 = affine_map<(d0, d1) -> (d0, d1)>

func.func @store_only_transposed_transfer_write(%src: memref<2x4xf32>,
                                                %out: memref<4x2xf32>) {
  %out_t = memref.transpose %out (d0, d1) -> (d1, d0)
      : memref<4x2xf32> to memref<2x4xf32, strided<[1, 2]>>
  linalg.generic {
      indexing_maps = [#map2],
      iterator_types = ["parallel", "parallel"]
    } ins(%src : memref<2x4xf32>) {
  ^bb0(%value: f32):
    %i = linalg.index 0 : index
    %j = linalg.index 1 : index
    memref.store %value, %out_t[%i, %j]
      : memref<2x4xf32, strided<[1, 2]>>
    linalg.yield
  }
  return
}

// CHECK-LABEL: func.func @store_only_transposed_transfer_write
// CHECK-NOT: linalg.generic
// CHECK-NOT: vector.scatter
// CHECK: memref.transpose
// CHECK-NOT: linalg.generic
// CHECK-NOT: vector.scatter
// CHECK: vector.transfer_write
// CHECK-NOT: vector.scatter
// CHECK-NOT: linalg.generic

// -----

#map1 = affine_map<(d0) -> (d0)>

func.func @mixed_store_and_yield_keeps_fallback(%offsets: memref<8xi32>,
                                                %values: memref<8xf32>,
                                                %out: memref<8xf32>,
                                                %side: memref<?xf32>) {
  linalg.generic {
      indexing_maps = [#map1, #map1, #map1],
      iterator_types = ["parallel"]
    } ins(%offsets, %values : memref<8xi32>, memref<8xf32>)
      outs(%out : memref<8xf32>) {
  ^bb0(%idx_i32: i32, %value: f32, %old: f32):
    %idx = arith.index_cast %idx_i32 : i32 to index
    memref.store %value, %side[%idx] : memref<?xf32>
    linalg.yield %value : f32
  }
  return
}

// CHECK-LABEL: func.func @mixed_store_and_yield_keeps_fallback
// CHECK-NOT: vector.scatter
// CHECK: memref.store
// CHECK-NOT: vector.scatter

// -----

#map1 = affine_map<(d0) -> (d0)>

func.func @store_only_sitofp_value_keeps_fallback(%offsets: memref<8xi32>,
                                                  %values: memref<8xi32>,
                                                  %out: memref<?xf32>) {
  linalg.generic {
      indexing_maps = [#map1, #map1],
      iterator_types = ["parallel"]
    } ins(%offsets, %values : memref<8xi32>, memref<8xi32>) {
  ^bb0(%idx_i32: i32, %value_i32: i32):
    %idx = arith.index_cast %idx_i32 : i32 to index
    %value = arith.sitofp %value_i32 : i32 to f32
    memref.store %value, %out[%idx] : memref<?xf32>
    linalg.yield
  }
  return
}

// CHECK-LABEL: func.func @store_only_sitofp_value_keeps_fallback
// CHECK-NOT: vector.scatter
// CHECK: arith.sitofp
// CHECK: memref.store
// CHECK-NOT: vector.scatter
