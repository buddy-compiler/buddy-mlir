// RUN: buddy-opt %s -lower-linalg-to-vir | FileCheck %s

module {
  // Transposed addition of two tensors. Note that only one dynamic dimension is
  // allowed, so the shape is fully static here.
  func.func @permute_input(%A: memref<8x4xf32>, %B: memref<4x8xf32>) {
    linalg.generic {
      indexing_maps = [affine_map<(i,j)->(j,i)>, affine_map<(i,j)->(i,j)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%A : memref<8x4xf32>) outs(%B : memref<4x8xf32>) {
      ^bb0(%a: f32, %b: f32):
        linalg.yield %a : f32
    }
    return
  }
}

// CHECK-LABEL: func.func @permute_input
// CHECK: vir.set_vl
// CHECK: memref.transpose %{{.*}} (d0, d1) -> (d1, d0)
// CHECK: memref.transpose %{{.*}} (d0, d1) -> (d0, d1)
// CHECK: vir.load
// CHECK: vir.store
