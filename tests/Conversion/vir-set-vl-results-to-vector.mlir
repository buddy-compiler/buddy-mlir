// RUN: buddy-opt %s -lower-vir-to-vector='vector-width=4' -canonicalize -cse \
// RUN: | FileCheck %s

// A vir.set_vl with results computes on whole vectors of its constant vl:
// 10 lanes lower to two vectors of 4 and two scalars. The reduce folds the
// pieces in order, its result is used in the region and yielded, and the
// loop carries every piece of its dynamic vector.
func.func @dot_then_scale(%a: memref<10xf32>, %b: memref<10xf32>,
                          %o: memref<10xf32>) -> f32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %vl = arith.constant 10 : index
  %z = arith.constant -0.0 : f32
  %r = vir.set_vl %vl : index {
    %x = vir.load %a[%c0] : memref<10xf32> -> !vir.vec<?xf32>
    %y = vir.load %b[%c0] : memref<10xf32> -> !vir.vec<?xf32>
    %p = arith.mulf %x, %y : !vir.vec<?xf32>
    %s = vir.reduce %p, %z {kind = "add", reassoc} : !vir.vec<?xf32>, f32 -> f32
    %sv = vir.broadcast %s : f32 -> !vir.vec<?xf32>
    %acc = scf.for %i = %c0 to %c3 step %c1 iter_args(%v = %x) -> (!vir.vec<?xf32>) {
      %n = vir.fma %v, %sv, %y : !vir.vec<?xf32>
      scf.yield %n : !vir.vec<?xf32>
    }
    vir.store %acc, %o[%c0] : !vir.vec<?xf32> -> memref<10xf32>
    vector.yield %s : f32
  } -> f32
  return %r : f32
}

// CHECK-LABEL: func.func @dot_then_scale
// CHECK-NOT: vir.
// CHECK-DAG: %[[X0:.*]] = vector.load %{{.*}}[%c0] : memref<10xf32>, vector<4xf32>
// CHECK-DAG: %[[X1:.*]] = vector.load %{{.*}}[%c4] : memref<10xf32>, vector<4xf32>
// CHECK-DAG: %[[X2:.*]] = memref.load %{{.*}}[%c8] : memref<10xf32>
// CHECK-DAG: %[[X3:.*]] = memref.load %{{.*}}[%c9] : memref<10xf32>
// CHECK: %[[R0:.*]] = vector.reduction <add>, %{{.*}}, %{{.*}} fastmath<reassoc> : vector<4xf32> into f32
// CHECK: %[[R1:.*]] = vector.reduction <add>, %{{.*}}, %[[R0]] fastmath<reassoc> : vector<4xf32> into f32
// CHECK: %[[R2:.*]] = arith.addf %[[R1]], %{{.*}} : f32
// CHECK: %[[S:.*]] = arith.addf %[[R2]], %{{.*}} : f32
// CHECK: %[[SV:.*]] = vector.broadcast %[[S]] : f32 to vector<4xf32>
// CHECK: %[[L:.*]]:4 = scf.for {{.*}} iter_args(%{{.*}} = %[[X0]], %{{.*}} = %[[X1]], %{{.*}} = %[[X2]], %{{.*}} = %[[X3]]) -> (vector<4xf32>, vector<4xf32>, f32, f32)
// CHECK: vector.fma %{{.*}}, %[[SV]], %{{.*}} : vector<4xf32>
// CHECK: vector.fma %{{.*}}, %[[SV]], %{{.*}} : vector<4xf32>
// CHECK: math.fma %{{.*}}, %[[S]], %{{.*}} : f32
// CHECK: math.fma %{{.*}}, %[[S]], %{{.*}} : f32
// CHECK: scf.yield
// CHECK: vector.store %[[L]]#0, %{{.*}}[%c0]
// CHECK: vector.store %[[L]]#1, %{{.*}}[%c4]
// CHECK: memref.store %[[L]]#2, %{{.*}}[%c8]
// CHECK: memref.store %[[L]]#3, %{{.*}}[%c9]
// CHECK: return %[[S]] : f32

// vir.from_vector: a vector of vl lanes computed outside the region, split
// into the pieces.
func.func @from_vector(%v: vector<10xf32>, %b: memref<10xf32>) -> f32 {
  %c0 = arith.constant 0 : index
  %vl = arith.constant 10 : index
  %z = arith.constant -0.0 : f32
  %r = vir.set_vl %vl : index {
    %x = vir.from_vector %v : vector<10xf32> -> !vir.vec<?xf32>
    %y = vir.load %b[%c0] : memref<10xf32> -> !vir.vec<?xf32>
    %p = arith.mulf %x, %y : !vir.vec<?xf32>
    %s = vir.reduce %p, %z {kind = "add", reassoc} : !vir.vec<?xf32>, f32 -> f32
    vector.yield %s : f32
  } -> f32
  return %r : f32
}

// CHECK-LABEL: func.func @from_vector
// CHECK-NOT: vir.
// CHECK-DAG: vector.extract_strided_slice %{{.*}} offsets = [0], sizes = [4], strides = [1] : vector<10xf32> to vector<4xf32>
// CHECK-DAG: vector.extract_strided_slice %{{.*}} offsets = [4], sizes = [4], strides = [1] : vector<10xf32> to vector<4xf32>
// CHECK-DAG: vector.extract %{{.*}}[8] : f32 from vector<10xf32>
// CHECK-DAG: vector.extract %{{.*}}[9] : f32 from vector<10xf32>

// A loop keeps its comparison: an unsigned scf.for stays unsigned.
func.func @unsigned_loop(%lb: i8, %ub: i8, %step: i8) -> f32 {
  %vl = arith.constant 4 : index
  %zero = arith.constant 0.0 : f32
  %one = arith.constant 1.0 : f32
  %r = vir.set_vl %vl : index {
    %x = vir.broadcast %zero : f32 -> !vir.vec<?xf32>
    %ones = vir.broadcast %one : f32 -> !vir.vec<?xf32>
    %v = scf.for unsigned %i = %lb to %ub step %step
        iter_args(%a = %x) -> (!vir.vec<?xf32>) : i8 {
      %n = arith.addf %a, %ones : !vir.vec<?xf32>
      scf.yield %n : !vir.vec<?xf32>
    }
    %s = vir.reduce %v, %zero {kind = "add"} : !vir.vec<?xf32>, f32 -> f32
    vector.yield %s : f32
  } -> f32
  return %r : f32
}

// CHECK-LABEL: func.func @unsigned_loop
// CHECK: scf.for unsigned %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %{{.*}}) -> (vector<4xf32>) : i8
