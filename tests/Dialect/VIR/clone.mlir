// RUN: buddy-opt %s --allow-unregistered-dialect | FileCheck %s

// A dynamic vector type cloned with another element type (ShapedType::clone,
// e.g. the i1 result type arith.cmpf infers) keeps its scaling factor: it
// has as many lanes.

// CHECK-LABEL: func.func @cmp
// CHECK: %[[C:.*]] = arith.cmpf olt, %{{.*}}, %{{.*}} : !vir.vec<?xf32, m2>
// CHECK: "test.use"(%[[C]]) : (!vir.vec<?xi1, m2>) -> ()
// CHECK: %[[D:.*]] = arith.cmpf olt, %{{.*}}, %{{.*}} : !vir.vec<2x?xf32, f4>
// CHECK: "test.use"(%[[D]]) : (!vir.vec<2x?xi1, f4>) -> ()
func.func @cmp(%a: !vir.vec<?xf32, m2>, %b: !vir.vec<2x?xf32, f4>) {
  %c = arith.cmpf olt, %a, %a : !vir.vec<?xf32, m2>
  "test.use"(%c) : (!vir.vec<?xi1, m2>) -> ()
  %d = arith.cmpf olt, %b, %b : !vir.vec<2x?xf32, f4>
  "test.use"(%d) : (!vir.vec<2x?xi1, f4>) -> ()
  return
}
