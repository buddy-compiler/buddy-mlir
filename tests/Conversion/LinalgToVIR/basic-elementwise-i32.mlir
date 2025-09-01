// RUN: buddy-opt %s -lower-linalg-to-vir | FileCheck %s

module {
  // Basic elementwise addition on i32.
  func.func @eltwise_add_i32(%A: memref<4x?xi32>, %B: memref<4x?xi32>, %C: memref<4x?xi32>) {
    linalg.generic {
      indexing_maps = [affine_map<(i,j)->(i,j)>, affine_map<(i,j)->(i,j)>, affine_map<(i,j)->(i,j)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%A, %B : memref<4x?xi32>, memref<4x?xi32>) outs(%C : memref<4x?xi32>) {
      ^bb0(%a: i32, %b: i32, %c: i32):
        %add = arith.addi %a, %b : i32
        linalg.yield %add : i32
    }
    return
  }
}

// CHECK-LABEL: func.func @eltwise_add_i32
// CHECK: vir.set_vl
// CHECK: vir.load {{.*}} : memref<4x?xi32, strided<
// CHECK: vir.load {{.*}} : memref<4x?xi32, strided<
// CHECK: arith.addi
// CHECK: vir.store
