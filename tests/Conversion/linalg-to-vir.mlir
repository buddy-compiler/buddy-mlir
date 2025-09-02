// RUN: buddy-opt %s -lower-linalg-to-vir | FileCheck %s

// ----
// Basic elementwise addition on f32.
func.func @eltwise_add(%A: memref<1024x?xf32>, %B: memref<1024x?xf32>, %C: memref<1024x?xf32>) {
  linalg.generic
      { indexing_maps = [affine_map<(i,j)->(i,j)>, affine_map<(i,j)->(i,j)>, affine_map<(i,j)->(i,j)>],
        iterator_types = ["parallel", "parallel"] }
      ins(%A, %B : memref<1024x?xf32>, memref<1024x?xf32>)
      outs(%C : memref<1024x?xf32>) {
    ^bb0(%a: f32, %b: f32, %c: f32):
      %add = arith.addf %a, %b : f32
      linalg.yield %add : f32
  }
  return
}

// CHECK-LABEL: func.func @eltwise_add
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[C1:.*]] = arith.constant 1 : index
// CHECK: %[[VL:.*]] = memref.dim %arg0, %[[C1]] : memref<1024x?xf32>
// CHECK: vir.set_vl %[[VL]] : index {
// CHECK:   %[[TA:.*]] = memref.transpose %arg0 (d0, d1) -> (d0, d1) : memref<1024x?xf32> to memref<1024x?xf32, strided<[?, 1]>>
// CHECK:   %[[SA:.*]] = memref.subview %[[TA]][0, 0] [1024, %[[VL]]] [1, 1] : memref<1024x?xf32, strided<[?, 1]>> to memref<1024x?xf32, strided<[?, 1]>>
// CHECK:   %[[TB:.*]] = memref.transpose %arg1 (d0, d1) -> (d0, d1) : memref<1024x?xf32> to memref<1024x?xf32, strided<[?, 1]>>
// CHECK:   %[[SB:.*]] = memref.subview %[[TB]][0, 0] [1024, %[[VL]]] [1, 1] : memref<1024x?xf32, strided<[?, 1]>> to memref<1024x?xf32, strided<[?, 1]>>
// CHECK:   %[[TC:.*]] = memref.transpose %arg2 (d0, d1) -> (d0, d1) : memref<1024x?xf32> to memref<1024x?xf32, strided<[?, 1]>>
// CHECK:   %[[VA:.*]] = vir.load %[[SA]][] : memref<1024x?xf32, strided<[?, 1]>> -> !vir.vec<1024x?xf32>
// CHECK:   %[[VB:.*]] = vir.load %[[SB]][] : memref<1024x?xf32, strided<[?, 1]>> -> !vir.vec<1024x?xf32>
// CHECK:   %[[ADD:.*]] = arith.addf %[[VA]], %[[VB]] : !vir.vec<1024x?xf32>
// CHECK:   vir.store %[[ADD]], %[[TC]][%[[C0]], %[[C0]]] : !vir.vec<1024x?xf32> -> memref<1024x?xf32, strided<[?, 1]>>
// CHECK:   vector.yield
// CHECK: }

// ----
// Basic elementwise addition on i32.
func.func @eltwise_add_i32(%A: memref<4x?xi32>, %B: memref<4x?xi32>, %C: memref<4x?xi32>) {
  linalg.generic
      { indexing_maps = [affine_map<(i,j)->(i,j)>, affine_map<(i,j)->(i,j)>, affine_map<(i,j)->(i,j)>],
        iterator_types = ["parallel", "parallel"] }
      ins(%A, %B : memref<4x?xi32>, memref<4x?xi32>)
      outs(%C : memref<4x?xi32>) {
    ^bb0(%a: i32, %b: i32, %c: i32):
      %add = arith.addi %a, %b : i32
      linalg.yield %add : i32
  }
  return
}

// CHECK-LABEL: func.func @eltwise_add_i32
// CHECK: %[[C0_I:.*]] = arith.constant 0 : index
// CHECK: %[[C1_I:.*]] = arith.constant 1 : index
// CHECK: %[[VL_I32:.*]] = memref.dim %arg0, %[[C1_I]] : memref<4x?xi32>
// CHECK: vir.set_vl %[[VL_I32]] : index {
// CHECK:   %[[TA:.*]] = memref.transpose %arg0 (d0, d1) -> (d0, d1) : memref<4x?xi32> to memref<4x?xi32, strided<[?, 1]>>
// CHECK:   %[[SA:.*]] = memref.subview %[[TA]][0, 0] [4, %[[VL_I32]]] [1, 1] : memref<4x?xi32, strided<[?, 1]>> to memref<4x?xi32, strided<[?, 1]>>
// CHECK:   %[[TB:.*]] = memref.transpose %arg1 (d0, d1) -> (d0, d1) : memref<4x?xi32> to memref<4x?xi32, strided<[?, 1]>>
// CHECK:   %[[SB:.*]] = memref.subview %[[TB]][0, 0] [4, %[[VL_I32]]] [1, 1] : memref<4x?xi32, strided<[?, 1]>> to memref<4x?xi32, strided<[?, 1]>>
// CHECK:   %[[TC:.*]] = memref.transpose %arg2 (d0, d1) -> (d0, d1) : memref<4x?xi32> to memref<4x?xi32, strided<[?, 1]>>
// CHECK:   %[[LA:.*]] = vir.load %[[SA]][] : memref<4x?xi32, strided<[?, 1]>> -> !vir.vec<4x?xi32>
// CHECK:   %[[LB:.*]] = vir.load %[[SB]][] : memref<4x?xi32, strided<[?, 1]>> -> !vir.vec<4x?xi32>
// CHECK:   %[[ADD_I32:.*]] = arith.addi %[[LA]], %[[LB]] : !vir.vec<4x?xi32>
// CHECK:   vir.store %[[ADD_I32]], %[[TC]][%[[C0_I]], %[[C0_I]]] : !vir.vec<4x?xi32> -> memref<4x?xi32, strided<[?, 1]>>
// CHECK:   vector.yield
// CHECK: }

// ----
// Scalar addition to a tensor.
func.func @scalar_plus_tensor(%A: memref<2x?xf32>, %C: memref<2x?xf32>) {
  %cst = arith.constant 3.0 : f32
  linalg.generic
      { indexing_maps = [affine_map<(i,j)->(i,j)>, affine_map<(i,j)->(i,j)>],
        iterator_types = ["parallel", "parallel"] }
      ins(%A : memref<2x?xf32>) outs(%C : memref<2x?xf32>) {
    ^bb0(%a: f32, %c: f32):
      %sum = arith.addf %a, %cst : f32
      linalg.yield %sum : f32
  }
  return
}

// CHECK-LABEL: func.func @scalar_plus_tensor
// CHECK: %[[C0_S:.*]] = arith.constant 0 : index
// CHECK: %[[C1_S:.*]] = arith.constant 1 : index
// CHECK: %[[VL_S:.*]] = memref.dim %arg0, %[[C1_S]] : memref<2x?xf32>
// CHECK: vir.set_vl %[[VL_S]] : index {
// CHECK:   %[[TA:.*]] = memref.transpose %arg0 (d0, d1) -> (d0, d1) : memref<2x?xf32> to memref<2x?xf32, strided<[?, 1]>>
// CHECK:   %[[SA:.*]] = memref.subview %[[TA]][0, 0] [2, %[[VL_S]]] [1, 1] : memref<2x?xf32, strided<[?, 1]>> to memref<2x?xf32, strided<[?, 1]>>
// CHECK:   %[[TC:.*]] = memref.transpose %arg1 (d0, d1) -> (d0, d1) : memref<2x?xf32> to memref<2x?xf32, strided<[?, 1]>>
// CHECK:   %[[V:.*]] = vir.load %[[SA]][] : memref<2x?xf32, strided<[?, 1]>> -> !vir.vec<2x?xf32>
// CHECK:   %[[BC:.*]] = vir.broadcast %{{.*}} : f32 -> !vir.vec<2x?xf32>
// CHECK:   %[[R:.*]] = arith.addf %[[V]], %[[BC]] : !vir.vec<2x?xf32>
// CHECK:   vir.store %[[R]], %[[TC]][%[[C0_S]], %[[C0_S]]] : !vir.vec<2x?xf32> -> memref<2x?xf32, strided<[?, 1]>>
// CHECK:   vector.yield
// CHECK: }

// ----
// Multiple elementwise ops chained on f32 tensors.
func.func @mul_add(%A: memref<16x?xf32>, %B: memref<16x?xf32>, %C: memref<16x?xf32>) {
  linalg.generic
      { indexing_maps = [affine_map<(i,j)->(i,j)>, affine_map<(i,j)->(i,j)>, affine_map<(i,j)->(i,j)>],
        iterator_types = ["parallel", "parallel"] }
      ins(%A, %B : memref<16x?xf32>, memref<16x?xf32>)
      outs(%C : memref<16x?xf32>) {
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

// CHECK-LABEL: func.func @mul_add
// CHECK: %[[C0_M:.*]] = arith.constant 0 : index
// CHECK: %[[C1_M:.*]] = arith.constant 1 : index
// CHECK: %[[VL_M:.*]] = memref.dim %arg0, %[[C1_M]] : memref<16x?xf32>
// CHECK: vir.set_vl %[[VL_M]] : index {
// CHECK:   %[[TA:.*]] = memref.transpose %arg0 (d0, d1) -> (d0, d1) : memref<16x?xf32> to memref<16x?xf32, strided<[?, 1]>>
// CHECK:   %[[SA:.*]] = memref.subview %[[TA]][0, 0] [16, %[[VL_M]]] [1, 1] : memref<16x?xf32, strided<[?, 1]>> to memref<16x?xf32, strided<[?, 1]>>
// CHECK:   %[[TB:.*]] = memref.transpose %arg1 (d0, d1) -> (d0, d1) : memref<16x?xf32> to memref<16x?xf32, strided<[?, 1]>>
// CHECK:   %[[SB:.*]] = memref.subview %[[TB]][0, 0] [16, %[[VL_M]]] [1, 1] : memref<16x?xf32, strided<[?, 1]>> to memref<16x?xf32, strided<[?, 1]>>
// CHECK:   %[[TC:.*]] = memref.transpose %arg2 (d0, d1) -> (d0, d1) : memref<16x?xf32> to memref<16x?xf32, strided<[?, 1]>>
// CHECK:   %[[LA:.*]] = vir.load %[[SA]][] : memref<16x?xf32, strided<[?, 1]>> -> !vir.vec<16x?xf32>
// CHECK:   %[[LB:.*]] = vir.load %[[SB]][] : memref<16x?xf32, strided<[?, 1]>> -> !vir.vec<16x?xf32>
// CHECK:   %[[LC:.*]] = vir.load %[[TC]][] : memref<16x?xf32, strided<[?, 1]>> -> !vir.vec<16x?xf32>
// CHECK:   %[[T0:.*]] = arith.mulf %[[LA]], %[[LB]] : !vir.vec<16x?xf32>
// CHECK:   %[[T1:.*]] = arith.addf %[[T0]], %[[LC]] : !vir.vec<16x?xf32>
// CHECK:   %[[T2:.*]] = arith.subf %[[T1]], %[[LA]] : !vir.vec<16x?xf32>
// CHECK:   %[[T3:.*]] = arith.divf %[[T2]], %[[LB]] : !vir.vec<16x?xf32>
// CHECK:   %[[T4:.*]] = arith.negf %[[T3]] : !vir.vec<16x?xf32>
// CHECK:   %[[BC:.*]] = vir.broadcast %{{.*}} : f32 -> !vir.vec<16x?xf32>
// CHECK:   %[[T5:.*]] = arith.mulf %[[T4]], %[[BC]] : !vir.vec<16x?xf32>
// CHECK:   %[[T6:.*]] = arith.addf %[[T5]], %[[T0]] : !vir.vec<16x?xf32>
// CHECK:   vir.store %[[T6]], %[[TC]][%[[C0_M]], %[[C0_M]]] : !vir.vec<16x?xf32> -> memref<16x?xf32, strided<[?, 1]>>
// CHECK:   vector.yield
// CHECK: }

// ----
// Transposed addition of two tensors. Only one dynamic dimension supported,
// so this shape is fully static.
func.func @permute_input(%A: memref<8x4xf32>, %B: memref<4x8xf32>) {
  linalg.generic
      { indexing_maps = [affine_map<(i,j)->(j,i)>, affine_map<(i,j)->(i,j)>],
        iterator_types = ["parallel", "parallel"] }
      ins(%A : memref<8x4xf32>) outs(%B : memref<4x8xf32>) {
    ^bb0(%a: f32, %b: f32):
      linalg.yield %a : f32
  }
  return
}

// CHECK-LABEL: func.func @permute_input
// CHECK: %[[C0_P:.*]] = arith.constant 0 : index
// CHECK: %[[C8_P:.*]] = arith.constant 8 : index
// CHECK: vir.set_vl %[[C8_P]] : index {
// CHECK:   %[[TA:.*]] = memref.transpose %arg0 (d0, d1) -> (d1, d0) : memref<8x4xf32> to memref<4x8xf32, strided<[1, 4]>>
// CHECK:   %[[TB:.*]] = memref.transpose %arg1 (d0, d1) -> (d0, d1) : memref<4x8xf32> to memref<4x8xf32, strided<[8, 1]>>
// CHECK:   %[[VA:.*]] = vir.load %[[TA]][] : memref<4x8xf32, strided<[1, 4]>> -> !vir.vec<4x8xf32>
// CHECK:   vir.store %[[VA]], %[[TB]][%[[C0_P]], %[[C0_P]]] : !vir.vec<4x8xf32> -> memref<4x8xf32, strided<[8, 1]>>
// CHECK:   vector.yield
// CHECK: }

// ----
// Initialize a constant tensor.
func.func @constant_init(%C: memref<16x?xf32>) {
  %cst = arith.constant 2.0 : f32
  %cst1 = arith.constant 3.0 : f32
  linalg.generic
      { indexing_maps = [affine_map<(i,j)->(i,j)>],
        iterator_types = ["parallel", "parallel"] }
      outs(%C : memref<16x?xf32>) {
    ^bb0(%c: f32):
      %v = arith.addf %cst, %cst1 : f32
      linalg.yield %v : f32
  }
  return
}

// CHECK-LABEL: func.func @constant_init
// CHECK: %[[C0_C:.*]] = arith.constant 0 : index
// CHECK: %[[C1_C:.*]] = arith.constant 1 : index
// CHECK: %[[VL_C:.*]] = memref.dim %arg0, %[[C1_C]] : memref<16x?xf32>
// CHECK: vir.set_vl %[[VL_C]] : index {
// CHECK:   %[[T:.*]] = memref.transpose %arg0 (d0, d1) -> (d0, d1) : memref<16x?xf32> to memref<16x?xf32, strided<[?, 1]>>
// CHECK:   %[[BC0:.*]] = vir.broadcast %{{.*}} : f32 -> !vir.vec<16x?xf32>
// CHECK:   %[[BC1:.*]] = vir.broadcast %{{.*}} : f32 -> !vir.vec<16x?xf32>
// CHECK:   %[[SUM:.*]] = arith.addf %[[BC0]], %[[BC1]] : !vir.vec<16x?xf32>
// CHECK:   vir.store %[[SUM]], %[[T]][%[[C0_C]], %[[C0_C]]] : !vir.vec<16x?xf32> -> memref<16x?xf32, strided<[?, 1]>>
// CHECK:   vector.yield
// CHECK: }

// ----
// Broadcast along the first loop dimension: input rank 1, loop rank 2.
func.func @broadcast_along_i(%A: memref<?xf32>, %B: memref<2x?xf32>) {
  linalg.generic
      { indexing_maps = [affine_map<(i,j)->(j)>, affine_map<(i,j)->(i,j)>],
        iterator_types = ["parallel", "parallel"] }
      ins(%A : memref<?xf32>) outs(%B : memref<2x?xf32>) {
    ^bb0(%a: f32, %b: f32):
      linalg.yield %a : f32
  }
  return
}

// CHECK-LABEL: func.func @broadcast_along_i
// CHECK: %[[C0_B:.*]] = arith.constant 0 : index
// CHECK: %[[VL_B:.*]] = memref.dim %arg0, %[[C0_B]] : memref<?xf32>
// CHECK: vir.set_vl %[[VL_B]] : index {
// CHECK:   %[[DIM:.*]] = memref.dim %arg0, %[[C0_B]] : memref<?xf32>
// CHECK:   %[[EXP:.*]] = memref.expand_shape %arg0 {{\[\[}}0, 1]] output_shape [1, %[[DIM]]] : memref<?xf32> into memref<1x?xf32>
// CHECK:   %[[T0:.*]] = memref.transpose %[[EXP]] (d0, d1) -> (d0, d1) : memref<1x?xf32> to memref<1x?xf32, strided<[?, 1]>>
// CHECK:   %[[SV:.*]] = memref.subview %[[T0]][0, 0] [2, %[[VL_B]]] [0, 1] : memref<1x?xf32, strided<[?, 1]>> to memref<2x?xf32, strided<[0, 1]>>
// CHECK:   %[[T1:.*]] = memref.transpose %arg1 (d0, d1) -> (d0, d1) : memref<2x?xf32> to memref<2x?xf32, strided<[?, 1]>>
// CHECK:   %[[V:.*]] = vir.load %[[SV]][] : memref<2x?xf32, strided<[0, 1]>> -> !vir.vec<2x?xf32>
// CHECK:   vir.store %[[V]], %[[T1]][%[[C0_B]], %[[C0_B]]] : !vir.vec<2x?xf32> -> memref<2x?xf32, strided<[?, 1]>>
// CHECK:   vector.yield
// CHECK: }
