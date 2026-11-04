// RUN: buddy-opt %s -lower-vir-to-vector='rvv-vlen=128 rvv-vlen-exact=false' \
// RUN: | FileCheck %s

// With only the minimum VLEN (rvv-vlen-exact=false), a vir.set_vl with
// results on fixed vectors is lowered: fixed vectors are valid on any VLEN.
func.func @dot(%a: memref<64xf32>) -> f32 {
  %vl = arith.constant 64 : index
  %z = arith.constant 0.0 : f32
  %r = vir.set_vl %vl : index {
    %x = vir.load %a[] : memref<64xf32> -> !vir.vec<?xf32>
    %s = vir.reduce %x, %z {kind = "add"} : !vir.vec<?xf32>, f32 -> f32
    vector.yield %s : f32
  } -> f32
  return %r : f32
}

// CHECK-LABEL: func.func @dot
// CHECK: vector.reduction <add>, %{{.*}}, %{{.*}} : vector<{{[0-9]+}}xf32> into f32
