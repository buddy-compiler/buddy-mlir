// RUN: buddy-opt %s -lower-vir-to-vector='rvv-vlen=1024' | FileCheck %s
// RUN: buddy-opt %s -lower-vir-to-vector | FileCheck %s

// A vir.shuffle of fixed vectors takes any number of lanes: one
// vector<48xf32> for 2 rows of 24 (on scalable vectors, N * vscale lanes
// with N a power of two).
func.func @rows24(%a: memref<2x24xf32>) -> f32 {
  %vl = arith.constant 24 : index
  %z = arith.constant 0.0 : f32
  %r = vir.set_vl %vl : index {
    %x = vir.load %a[] : memref<2x24xf32> -> !vir.vec<2x?xf32>
    %y = vir.shuffle %x, %x [1, 0] : !vir.vec<2x?xf32>
    %s = vir.reduce %y, %z {kind = "add"} : !vir.vec<2x?xf32>, f32 -> f32
    vector.yield %s : f32
  } -> f32
  return %r : f32
}

// CHECK-LABEL: func.func @rows24
// CHECK: %[[X:.*]] = vector.load {{.*}} : memref<48xf32>, vector<48xf32>
// CHECK: vector.shuffle %[[X]], %[[X]] [24, 25, {{.*}}, 46, 47, 0, 1, {{.*}}, 22, 23] : vector<48xf32>, vector<48xf32>
