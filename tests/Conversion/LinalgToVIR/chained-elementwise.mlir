// RUN: buddy-opt %s -lower-linalg-to-vir | FileCheck %s

module {
  // Multiple elementwise ops chained on f32 tensors:
  func.func @mul_add(%A: memref<16x?xf32>, %B: memref<16x?xf32>, %C: memref<16x?xf32>) {
    linalg.generic {
      indexing_maps = [affine_map<(i,j)->(i,j)>, affine_map<(i,j)->(i,j)>, affine_map<(i,j)->(i,j)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%A, %B : memref<16x?xf32>, memref<16x?xf32>) outs(%C : memref<16x?xf32>) {
      ^bb0(%a: f32, %b: f32, %c: f32):
        %t0 = arith.mulf %a, %b : f32
        %t1 = arith.addf %t0, %c : f32
        %t2 = arith.subf %t1, %a : f32
        %t3 = arith.divf %t2, %b : f32
        %t4 = arith.negf %t3 : f32
        %cst = arith.constant 5.000000e-01 : f32
        %t5 = arith.mulf %t4, %cst : f32
        %t6 = arith.addf %t5, %t0 : f32
        linalg.yield %t6 : f32
    }
    return
  }
}

// CHECK-LABEL: func.func @mul_add
// CHECK: vir.set_vl
// CHECK: vir.load {{.*}} : memref<16x?xf32, strided<
// CHECK: vir.load {{.*}} : memref<16x?xf32, strided<
// CHECK: vir.load {{.*}} : memref<16x?xf32, strided<
// CHECK: arith.mulf
// CHECK: arith.addf
// CHECK: arith.subf
// CHECK: arith.divf
// CHECK: arith.negf
// CHECK: vir.store
