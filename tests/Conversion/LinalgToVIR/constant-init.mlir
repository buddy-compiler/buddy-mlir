// RUN: buddy-opt %s -lower-linalg-to-vir | FileCheck %s

module {
  // Initialize a constant tensor.
  func.func @constant_init(%C: memref<16x?xf32>) {
    %cst = arith.constant 2.0 : f32
    %cst1 = arith.constant 3.0 : f32
    linalg.generic {
      indexing_maps = [affine_map<(i,j)->(i,j)>],
      iterator_types = ["parallel", "parallel"]
    } outs(%C : memref<16x?xf32>) {
      ^bb0(%c: f32):
        %v = arith.addf %cst, %cst1 : f32
        linalg.yield %v : f32
    }
    return
  }
}

// CHECK-LABEL: func.func @constant_init
// CHECK: vir.set_vl
// CHECK: memref.transpose {{.*}} : memref<16x?xf32> to memref<16x?xf32, strided<
// CHECK: vir.broadcast %{{.*}} : f32 -> !vir.vec<16x?xf32>
// CHECK: vir.broadcast %{{.*}} : f32 -> !vir.vec<16x?xf32>
// CHECK: arith.addf
// CHECK: vir.store
