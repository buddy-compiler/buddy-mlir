// RUN: buddy-opt %s -split-input-file -cse -canonicalize | FileCheck %s

// The value ops of VIR have no memory effects (Pure): two of the same
// operands are one (-cse), each result stored to its own memref; the unused
// ones are erased. The stores are kept.

// CHECK-LABEL: func.func @cse
// CHECK: vir.set_vl
// CHECK-NEXT: %[[S:.*]] = vir.broadcast
// CHECK-NEXT: %[[K:.*]] = vir.constant
// CHECK-NEXT: %[[X:.*]] = vir.load
// CHECK-NEXT: %[[Y:.*]] = vir.fma %[[X]], %[[S]], %[[K]]
// CHECK-NEXT: vir.store %[[Y]], %arg1
// CHECK-NEXT: vir.store %[[Y]], %arg2
// CHECK-NEXT: %[[H:.*]] = vir.truncf %[[X]]
// CHECK-NEXT: vir.store %[[H]], %arg3
// CHECK-NEXT: vir.store %[[H]], %arg3
// CHECK-NEXT: %[[E:.*]] = vir.extf %[[H]]
// CHECK-NEXT: vir.store %[[E]], %arg1
// CHECK-NEXT: vir.store %[[E]], %arg2
// CHECK-NEXT: %[[M:.*]] = arith.cmpf olt, %[[X]], %[[K]]
// CHECK-NEXT: %[[W:.*]] = vir.select %[[M]], %[[X]], %[[K]]
// CHECK-NEXT: vir.store %[[W]], %arg1
// CHECK-NEXT: vir.store %[[W]], %arg2
// CHECK-NEXT: vector.yield
func.func @cse(%a: memref<256xf32>, %o0: memref<256xf32>,
               %o1: memref<256xf32>, %h: memref<256xf16>, %f: f32) {
  %c0 = arith.constant 0 : index
  %vl = arith.constant 256 : index
  vir.set_vl %vl : index {
    %s0 = vir.broadcast %f : f32 -> !vir.vec<?xf32>
    %s1 = vir.broadcast %f : f32 -> !vir.vec<?xf32>
    %k0 = vir.constant {value = 1.0 : f32} : !vir.vec<?xf32>
    %k1 = vir.constant {value = 1.0 : f32} : !vir.vec<?xf32>
    %x = vir.load %a[%c0] : memref<256xf32> -> !vir.vec<?xf32>
    %y0 = vir.fma %x, %s0, %k0 : !vir.vec<?xf32>
    %y1 = vir.fma %x, %s1, %k1 : !vir.vec<?xf32>
    vir.store %y0, %o0[%c0] : !vir.vec<?xf32> -> memref<256xf32>
    vir.store %y1, %o1[%c0] : !vir.vec<?xf32> -> memref<256xf32>
    %h0 = vir.truncf %x : !vir.vec<?xf32> -> !vir.vec<?xf16>
    %h1 = vir.truncf %x : !vir.vec<?xf32> -> !vir.vec<?xf16>
    vir.store %h0, %h[%c0] : !vir.vec<?xf16> -> memref<256xf16>
    vir.store %h1, %h[%c0] : !vir.vec<?xf16> -> memref<256xf16>
    %e0 = vir.extf %h0 : !vir.vec<?xf16> -> !vir.vec<?xf32>
    %e1 = vir.extf %h1 : !vir.vec<?xf16> -> !vir.vec<?xf32>
    vir.store %e0, %o0[%c0] : !vir.vec<?xf32> -> memref<256xf32>
    vir.store %e1, %o1[%c0] : !vir.vec<?xf32> -> memref<256xf32>
    %m = arith.cmpf olt, %x, %k0 : !vir.vec<?xf32>
    %w0 = vir.select %m, %x, %k0 : !vir.vec<?xi1>, !vir.vec<?xf32>
    %w1 = vir.select %m, %x, %k1 : !vir.vec<?xi1>, !vir.vec<?xf32>
    vir.store %w0, %o0[%c0] : !vir.vec<?xf32> -> memref<256xf32>
    vir.store %w1, %o1[%c0] : !vir.vec<?xf32> -> memref<256xf32>
    vector.yield
  }
  return
}

// -----

// The integer extends.
// CHECK-LABEL: func.func @cse_ext
// CHECK: vir.set_vl
// CHECK-NEXT: %[[X:.*]] = vir.load
// CHECK-NEXT: %[[S:.*]] = vir.extsi %[[X]]
// CHECK-NEXT: vir.store %[[S]], %arg1
// CHECK-NEXT: vir.store %[[S]], %arg2
// CHECK-NEXT: %[[U:.*]] = vir.extui %[[X]]
// CHECK-NEXT: vir.store %[[U]], %arg1
// CHECK-NEXT: vir.store %[[U]], %arg2
// CHECK-NEXT: vector.yield
func.func @cse_ext(%a: memref<256xi8>, %o0: memref<256xi32>,
                   %o1: memref<256xi32>) {
  %c0 = arith.constant 0 : index
  %vl = arith.constant 256 : index
  vir.set_vl %vl : index {
    %x = vir.load %a[%c0] : memref<256xi8> -> !vir.vec<?xi8>
    %s0 = vir.extsi %x : !vir.vec<?xi8> -> !vir.vec<?xi32>
    %s1 = vir.extsi %x : !vir.vec<?xi8> -> !vir.vec<?xi32>
    vir.store %s0, %o0[%c0] : !vir.vec<?xi32> -> memref<256xi32>
    vir.store %s1, %o1[%c0] : !vir.vec<?xi32> -> memref<256xi32>
    %u0 = vir.extui %x : !vir.vec<?xi8> -> !vir.vec<?xi32>
    %u1 = vir.extui %x : !vir.vec<?xi8> -> !vir.vec<?xi32>
    vir.store %u0, %o0[%c0] : !vir.vec<?xi32> -> memref<256xi32>
    vir.store %u1, %o1[%c0] : !vir.vec<?xi32> -> memref<256xi32>
    vector.yield
  }
  return
}

// -----

// vir.from_vector and vir.reduce, in a vir.set_vl with results.
// CHECK-LABEL: func.func @cse_from_vector
// CHECK: vir.set_vl
// CHECK-NEXT: %[[V:.*]] = vir.from_vector
// CHECK-NEXT: %[[R:.*]] = vir.reduce %[[V]]
// CHECK-NEXT: %[[S:.*]] = arith.addf %[[R]], %[[R]]
// CHECK-NEXT: vector.yield %[[S]]
func.func @cse_from_vector(%v: vector<4xf32>) -> f32 {
  %z = arith.constant 0.0 : f32
  %vl = arith.constant 4 : index
  %r = vir.set_vl %vl : index {
    %a = vir.from_vector %v : vector<4xf32> -> !vir.vec<?xf32>
    %b = vir.from_vector %v : vector<4xf32> -> !vir.vec<?xf32>
    %ra = vir.reduce %a, %z {kind = "add"} : !vir.vec<?xf32>, f32 -> f32
    %rb = vir.reduce %b, %z {kind = "add"} : !vir.vec<?xf32>, f32 -> f32
    %s = arith.addf %ra, %rb : f32
    vector.yield %s : f32
  } -> f32
  return %r : f32
}

// -----

// Unused, each is erased; the load too (it only reads). The store is kept.
// CHECK-LABEL: func.func @dce
// CHECK: vir.set_vl
// CHECK-NEXT: %[[K:.*]] = vir.constant
// CHECK-NEXT: vir.store %[[K]]
// CHECK-NEXT: vector.yield
func.func @dce(%a: memref<256xf32>, %b: memref<256xi8>, %o: memref<256xf32>,
               %f: f32) {
  %c0 = arith.constant 0 : index
  %z = arith.constant 0.0 : f32
  %vl = arith.constant 256 : index
  vir.set_vl %vl : index {
    %s = vir.broadcast %f : f32 -> !vir.vec<?xf32>
    %k = vir.constant {value = 1.0 : f32} : !vir.vec<?xf32>
    %k2 = vir.constant {value = 2.0 : f32} : !vir.vec<?xf32>
    %x = vir.load %a[%c0] : memref<256xf32> -> !vir.vec<?xf32>
    %y = vir.fma %x, %s, %k2 : !vir.vec<?xf32>
    %h = vir.truncf %y : !vir.vec<?xf32> -> !vir.vec<?xf16>
    %e = vir.extf %h : !vir.vec<?xf16> -> !vir.vec<?xf32>
    %m = arith.cmpf olt, %x, %k : !vir.vec<?xf32>
    %w = vir.select %m, %x, %k : !vir.vec<?xi1>, !vir.vec<?xf32>
    %r = vir.reduce %w, %z {kind = "add"} : !vir.vec<?xf32>, f32 -> f32
    %i = vir.load %b[%c0] : memref<256xi8> -> !vir.vec<?xi8>
    %is = vir.extsi %i : !vir.vec<?xi8> -> !vir.vec<?xi32>
    %iu = vir.extui %i : !vir.vec<?xi8> -> !vir.vec<?xi32>
    vir.store %k, %o[%c0] : !vir.vec<?xf32> -> memref<256xf32>
    vector.yield
  }
  return
}

// -----

// An unused vir.from_vector, in a vir.set_vl with results.
// CHECK-LABEL: func.func @dce_from_vector
// CHECK: vir.set_vl
// CHECK-NEXT: %[[V:.*]] = vir.from_vector %arg0
// CHECK-NEXT: %[[R:.*]] = vir.reduce %[[V]]
// CHECK-NEXT: vector.yield %[[R]]
func.func @dce_from_vector(%v: vector<4xf32>, %w: vector<4xf32>) -> f32 {
  %z = arith.constant 0.0 : f32
  %vl = arith.constant 4 : index
  %r = vir.set_vl %vl : index {
    %a = vir.from_vector %v : vector<4xf32> -> !vir.vec<?xf32>
    %dead = vir.from_vector %w : vector<4xf32> -> !vir.vec<?xf32>
    %ra = vir.reduce %a, %z {kind = "add"} : !vir.vec<?xf32>, f32 -> f32
    vector.yield %ra : f32
  } -> f32
  return %r : f32
}
