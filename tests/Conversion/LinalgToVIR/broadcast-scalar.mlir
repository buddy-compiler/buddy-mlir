// RUN: buddy-opt %s -lower-linalg-to-vir | FileCheck %s

module {
  // Scalar addition to a tensor.
  func.func @scalar_plus_tensor(%A: memref<2x?xf32>, %C: memref<2x?xf32>) {
    %cst = arith.constant 3.0 : f32
    linalg.generic {
      indexing_maps = [affine_map<(i,j)->(i,j)>, affine_map<(i,j)->(i,j)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%A : memref<2x?xf32>) outs(%C : memref<2x?xf32>) {
      ^bb0(%a: f32, %c: f32):
        %sum = arith.addf %a, %cst : f32
        linalg.yield %sum : f32
    }
    return
  }
}

// CHECK-LABEL: func.func @scalar_plus_tensor
// CHECK: vir.set_vl
// CHECK: memref.transpose {{.*}} : memref<2x?xf32> to memref<2x?xf32, strided<
// CHECK: vir.load {{.*}} : memref<2x?xf32, strided<
// CHECK: vir.broadcast %{{.*}} : f32 -> !vir.vec<2x?xf32>
// CHECK: arith.addf
// CHECK: vir.store
