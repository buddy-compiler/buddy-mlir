// RUN: buddy-opt %s -cse -canonicalize -lower-vir-to-vector='vector-width=4' \
// RUN: | FileCheck %s

// A vir.set_vl region left with no dynamic vector, its unused vector ops
// erased (the VIR value ops are Pure): its ops, if any, are inlined in the
// parent block and its results are those of its yield.

// No result, nothing left.
// CHECK-LABEL: func.func @empty
// CHECK-NOT: vir.
// CHECK-NOT: vector.
// CHECK: return
func.func @empty(%f: f32) {
  %vl = arith.constant 4 : index
  vir.set_vl %vl : index {
    %dead = vir.broadcast %f : f32 -> !vir.vec<?xf32>
    vector.yield
  }
  return
}

// A scalar result, forwarded.
// CHECK-LABEL: func.func @scalar_result
// CHECK-SAME: (%[[F:.*]]: f32)
// CHECK-NOT: vir.
// CHECK: return %[[F]] : f32
func.func @scalar_result(%f: f32) -> f32 {
  %vl = arith.constant 4 : index
  %r = vir.set_vl %vl : index {
    %dead = vir.broadcast %f : f32 -> !vir.vec<?xf32>
    vector.yield %f : f32
  } -> f32
  return %r : f32
}

// Scalar ops left, one of them a store: they are kept, in order.
// CHECK-LABEL: func.func @scalar_ops
// CHECK-SAME: (%[[F:.*]]: f32, %[[M:.*]]: memref<4xf32>)
// CHECK-NOT: vir.
// CHECK: %[[S:.*]] = arith.addf %[[F]], %[[F]] : f32
// CHECK: memref.store %[[S]], %[[M]]
// CHECK: %[[P:.*]] = arith.mulf %[[S]], %[[F]] : f32
// CHECK: return %[[P]] : f32
func.func @scalar_ops(%f: f32, %m: memref<4xf32>) -> f32 {
  %c0 = arith.constant 0 : index
  %vl = arith.constant 4 : index
  %r = vir.set_vl %vl : index {
    %dead = vir.broadcast %f : f32 -> !vir.vec<?xf32>
    %s = arith.addf %f, %f : f32
    memref.store %s, %m[%c0] : memref<4xf32>
    %p = arith.mulf %s, %f : f32
    vector.yield %p : f32
  } -> f32
  return %r : f32
}
