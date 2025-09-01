// RUN: buddy-opt %s -lower-linalg-to-vir | FileCheck %s

module {
  // Basic elementwise addition on f32.
  func.func @eltwise_add(%A: memref<1024x?xf32>, %B: memref<1024x?xf32>, %C: memref<1024x?xf32>) {
    linalg.generic {
      indexing_maps = [affine_map<(i,j)->(i,j)>, affine_map<(i,j)->(i,j)>, affine_map<(i,j)->(i,j)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%A, %B : memref<1024x?xf32>, memref<1024x?xf32>) outs(%C : memref<1024x?xf32>) {
      ^bb0(%a: f32, %b: f32, %c: f32):
        %add = arith.addf %a, %b : f32
        linalg.yield %add : f32
    }
    return
  }
}

// CHECK-LABEL: func.func @eltwise_add
// CHECK: vir.set_vl
// CHECK: memref.transpose
// CHECK: memref.subview
// CHECK: vir.load {{.*}} : memref<1024x?xf32, strided<
// CHECK: vir.load {{.*}} : memref<1024x?xf32, strided<
// CHECK: arith.addf
// CHECK: vir.store
