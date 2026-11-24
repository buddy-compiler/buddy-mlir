// RUN: buddy-opt %s -cse -canonicalize | FileCheck %s

// The value ops of VIR have no memory effects (Pure): two of the same
// operands are one (-cse) and the unused ones are erased; the store is kept.

// CHECK-LABEL: func.func @pure
// CHECK: vir.set_vl
// CHECK-NEXT: %[[S:.*]] = vir.broadcast %{{.*}} : f32 -> !vir.vec<?xf32>
// CHECK-NEXT: %[[X:.*]] = vir.load
// CHECK-NEXT: %[[Y:.*]] = vir.fma %[[X]], %[[S]], %[[S]]
// CHECK-NEXT: vir.store %[[Y]]
// CHECK-NEXT: vector.yield
func.func @pure(%a: memref<256xf32>, %o: memref<256xf32>, %f: f32) {
  %c0 = arith.constant 0 : index
  %vl = arith.constant 256 : index
  vir.set_vl %vl : index {
    %s = vir.broadcast %f : f32 -> !vir.vec<?xf32>
    %t = vir.broadcast %f : f32 -> !vir.vec<?xf32>
    %x = vir.load %a[%c0] : memref<256xf32> -> !vir.vec<?xf32>
    %k = vir.constant {value = 1.0 : f32} : !vir.vec<?xf32>
    %y = vir.fma %x, %s, %t : !vir.vec<?xf32>
    %z = vir.fma %x, %s, %t : !vir.vec<?xf32>
    %h = vir.truncf %y : !vir.vec<?xf32> -> !vir.vec<?xf16>
    %e = vir.extf %h : !vir.vec<?xf16> -> !vir.vec<?xf32>
    %m = arith.cmpf olt, %x, %k : !vir.vec<?xf32>
    %w = vir.select %m, %x, %k : !vir.vec<?xi1>, !vir.vec<?xf32>
    %z0 = arith.constant 0.0 : f32
    %r = vir.reduce %w, %z0 {kind = "add"} : !vir.vec<?xf32>, f32 -> f32
    vir.store %z, %o[%c0] : !vir.vec<?xf32> -> memref<256xf32>
    vector.yield
  }
  return
}
