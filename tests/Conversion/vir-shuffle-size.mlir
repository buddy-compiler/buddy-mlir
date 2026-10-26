// RUN: buddy-opt %s -lower-vir-to-vector='use-scalable=true rvv-vlen=256 verbose=true' \
// RUN: | FileCheck %s

// With a vir.shuffle, each dynamic vector is one vector of all its lanes,
// whatever the register pressure: here the pressure gives LMUL 4 (two
// vector<[8]xf32> of 32 lanes at VLEN 256), the shuffle one vector<[16]xf32>
// of the 64 lanes (LMUL 8).
func.func @rope(%q: memref<2x32xf32>, %cs: memref<2x32xf32>,
                %sn: memref<2x32xf32>, %o: memref<2x32xf32>) -> f32 {
  %vl = arith.constant 32 : index
  %z = arith.constant 0.0 : f32
  %r = vir.set_vl %vl : index {
    %x = vir.load %q[] : memref<2x32xf32> -> !vir.vec<2x?xf32>
    %c = vir.load %cs[] : memref<2x32xf32> -> !vir.vec<2x?xf32>
    %s = vir.load %sn[] : memref<2x32xf32> -> !vir.vec<2x?xf32>
    %nx = arith.negf %x : !vir.vec<2x?xf32>
    %rot = vir.shuffle %nx, %x [1, 2] : !vir.vec<2x?xf32>
    %t = arith.mulf %rot, %s : !vir.vec<2x?xf32>
    %y = vir.fma %x, %c, %t : !vir.vec<2x?xf32>
    vir.store %y, %o[] : !vir.vec<2x?xf32> -> memref<2x32xf32>
    %d = vir.reduce %y, %z {kind = "add"} : !vir.vec<2x?xf32>, f32 -> f32
    vector.yield %d : f32
  } -> f32
  return %r : f32
}

// CHECK: Calculated LMUL = 4
// CHECK: vir.shuffle: one vector of 64 lanes, width 16
// CHECK-LABEL: func.func @rope
// CHECK-NOT: vector.create_mask
// CHECK: vector.create_mask %{{.*}} : vector<[16]xi1>
// CHECK: llvm.call_intrinsic "llvm.vector.splice.left"({{.*}}) : (vector<[16]xf32>, vector<[16]xf32>, i32) -> vector<[16]xf32>
