// RUN: buddy-opt %s -lower-vir-to-vector='use-scalable=true' | FileCheck %s

// use-scalable without rvv-vlen is for any RVV VLEN (vectors sized for the
// least, 128). A strip-mined region steps by N * vscale, read at run time:
// valid on any VLEN, lowered as for an exact VLEN.
func.func @scale(%a: memref<?xf32>, %n: index, %w: f32) {
  vir.set_vl %n : index {
    %x = vir.load %a[] : memref<?xf32> -> !vir.vec<?xf32>
    %y = arith.mulf %x, %x : !vir.vec<?xf32>
    vir.store %y, %a[] : !vir.vec<?xf32> -> memref<?xf32>
    vector.yield
  }
  return
}

// CHECK-LABEL: func.func @scale
// CHECK: %[[VS:.*]] = vector.vscale
// CHECK: %[[STEP:.*]] = arith.muli %{{.*}}, %[[VS]] : index
// CHECK: scf.for {{.*}} step %[[STEP]] {
// CHECK: vector.load {{.*}} : memref<?xf32>, vector<[{{[0-9]+}}]xf32>
