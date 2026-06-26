// RUN: buddy-opt %s -convert-linalg-to-tile | FileCheck %s

#map0 = affine_map<(batch, m, n, k) -> (batch, m, k)>
#map1 = affine_map<(batch, m, n, k) -> (batch, n, k)>
#map2 = affine_map<(batch, m, n, k) -> (batch, m, n)>
#map3 = affine_map<(m, n, k) -> (m, k)>
#map4 = affine_map<(m, n, k) -> (n, k)>
#map5 = affine_map<(m, n, k) -> (m, n)>

module {
  func.func @matmul_transpose_b(%A: memref<3x4xf32>,
                                %B: memref<5x4xf32>,
                                %C: memref<3x5xf32>) {
    linalg.matmul indexing_maps = [#map3, #map4, #map5]
      ins(%A, %B : memref<3x4xf32>, memref<5x4xf32>)
      outs(%C : memref<3x5xf32>)
    return
  }

  func.func @batch_matmul_transpose_b(%A: memref<2x3x4xf32>,
                                      %B: memref<2x5x4xf32>,
                                      %C: memref<2x3x5xf32>) {
    linalg.batch_matmul indexing_maps = [#map0, #map1, #map2]
      ins(%A, %B : memref<2x3x4xf32>, memref<2x5x4xf32>)
      outs(%C : memref<2x3x5xf32>)
    return
  }
}

// CHECK-LABEL: func.func @matmul_transpose_b
// CHECK:       memref.alloc() : memref<4x5xf32>
// CHECK:       scf.for
// CHECK:       scf.for
// CHECK:       memref.load {{.*}} : memref<5x4xf32>
// CHECK:       memref.store {{.*}} : memref<4x5xf32>
// CHECK:       tile.tile_matmul {{.*}} : memref<3x4xf32> memref<4x5xf32> memref<3x5xf32>
// CHECK:       memref.dealloc {{.*}} : memref<4x5xf32>

// CHECK-LABEL: func.func @batch_matmul_transpose_b
// CHECK:       memref.alloc() : memref<4x5xf32>
// CHECK:       scf.for
// CHECK:       scf.for
// CHECK:       memref.load {{.*}} : memref<5x4xf32
// CHECK:       memref.store {{.*}} : memref<4x5xf32>
// CHECK:       tile.tile_matmul {{.*}} : memref<3x4xf32{{.*}} memref<4x5xf32> memref<3x5xf32
// CHECK:       memref.dealloc {{.*}} : memref<4x5xf32>
