// RUN: buddy-opt %s -cse -lower-vir-to-vector='vector-width=4' \
// RUN: | FileCheck %s
// RUN: buddy-opt %s -cse -canonicalize -lower-vir-to-vector='vector-width=4' \
// RUN: | FileCheck %s

// A vir.set_vl region left with no dynamic vector, its unused vector ops
// erased (the VIR value ops are Pure), with -cse alone or after it
// -canonicalize:
// - without results, the region runs once per chunk of lanes, none for vl
//   0, and writes nothing but the vir.reduce results it stores (the
//   verifier): it is erased, its ops not moved where they may not have run;
// - with results (a constant vl), its scalar ops run once: they are
//   inlined in the parent block and its results are those of its yield.

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

// A read left (-cse erases the broadcast, not then the load it used), for a
// vl that may be 0, which runs the region never: the load is not run.
// CHECK-LABEL: func.func @zero_trip
// CHECK-NOT: memref.load
// CHECK-NOT: vir.
// CHECK: return
func.func @zero_trip(%m: memref<?xf32>, %vl: index) {
  %c0 = arith.constant 0 : index
  vir.set_vl %vl : index {
    %s = memref.load %m[%c0] : memref<?xf32>
    %dead = vir.broadcast %s : f32 -> !vir.vec<?xf32>
    vector.yield
  }
  return
}

// The same, of a constant vl 0.
// CHECK-LABEL: func.func @zero_vl
// CHECK-NOT: memref.load
// CHECK-NOT: vir.
// CHECK: return
func.func @zero_vl(%m: memref<0xf32>) {
  %c0 = arith.constant 0 : index
  %vl = arith.constant 0 : index
  vir.set_vl %vl : index {
    %s = memref.load %m[%c0] : memref<0xf32>
    %dead = vir.broadcast %s : f32 -> !vir.vec<?xf32>
    vector.yield
  }
  return
}
