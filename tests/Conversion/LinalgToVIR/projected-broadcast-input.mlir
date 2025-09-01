// RUN: buddy-opt %s -lower-linalg-to-vir | FileCheck %s

// Broadcast along the first loop dimension: input rank 1, loop rank 2.
// The pass expands shape, transposes (identity here), subviews with broadcast,
// loads via VIR, does elementwise yield, and stores.
module {
  func.func @broadcast_along_i(%A: memref<?xf32>, %B: memref<2x?xf32>) {
    linalg.generic {
      indexing_maps = [affine_map<(i,j)->(j)>, affine_map<(i,j)->(i,j)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%A : memref<?xf32>) outs(%B : memref<2x?xf32>) {
      ^bb0(%a: f32, %b: f32):
        linalg.yield %a : f32
    }
    return
  }
}

// CHECK-LABEL: func.func @broadcast_along_i
// CHECK: vir.set_vl
// CHECK: memref.expand_shape
// CHECK: memref.transpose
// CHECK: memref.subview
// CHECK: vir.load {{.*}} : memref<2x?xf32, strided<
// CHECK: vir.store
