// RUN: buddy-opt %s -lower-vir-to-vector='vector-width=4' -cse | FileCheck %s

// vir.reduce lowers to an ordered vector.reduction, and with `reassoc` to one
// whose lanes may be combined in any order.
func.func @vir_reduce(%arg0: memref<128xf32>, %out: memref<2xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c128 = arith.constant 128 : index
  vir.set_vl %c128 : index {
    %a = vir.load %arg0[] : memref<128xf32> -> !vir.vec<?xf32>
    %s0 = memref.load %out[%c0] : memref<2xf32>
    %r0 = vir.reduce %a, %s0 {kind = "add"} : !vir.vec<?xf32>, f32 -> f32
    memref.store %r0, %out[%c0] : memref<2xf32>
    %s1 = memref.load %out[%c1] : memref<2xf32>
    %r1 = vir.reduce %a, %s1 {kind = "add", reassoc} : !vir.vec<?xf32>, f32 -> f32
    memref.store %r1, %out[%c1] : memref<2xf32>
    vector.yield
  }
  return
}

// CHECK-LABEL: func.func @vir_reduce
// CHECK: vector.reduction <add>, %{{.*}}, %{{.*}} : vector<4xf32> into f32
// CHECK: vector.reduction <add>, %{{.*}}, %{{.*}} fastmath<reassoc> : vector<4xf32> into f32
