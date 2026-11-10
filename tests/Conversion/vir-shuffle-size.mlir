// RUN: buddy-opt %s -lower-vir-to-vector='use-scalable=true rvv-vlen=256 verbose=true' \
// RUN: | FileCheck %s

// A region with a vir.shuffle is sized as any: on scalable vectors every
// lane is known (vscale = rvv-vlen / 64), so the shuffle moves lanes between
// pieces. Here the pressure gives LMUL 4, two vector<[8]xf32> of 32 lanes at
// VLEN 256, one per row: rotate_half takes -x from the second piece and x
// from the first, no instruction.
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
// CHECK-LABEL: func.func @rope
// CHECK-NOT: splice
// CHECK-NOT: scalable.extract
// CHECK-NOT: mask
// CHECK: %[[X0:.*]] = vector.load %{{.*}} : memref<64xf32>, vector<[8]xf32>
// CHECK: %[[X1:.*]] = vector.load %{{.*}} : memref<64xf32>, vector<[8]xf32>
// CHECK: vector.load %{{.*}} : memref<64xf32>, vector<[8]xf32>
// CHECK: vector.load %{{.*}} : memref<64xf32>, vector<[8]xf32>
// CHECK: %[[S0:.*]] = vector.load %{{.*}} : memref<64xf32>, vector<[8]xf32>
// CHECK: %[[S1:.*]] = vector.load %{{.*}} : memref<64xf32>, vector<[8]xf32>
// CHECK: %[[N1:.*]] = arith.negf %[[X1]] : vector<[8]xf32>
// CHECK: arith.mulf %[[N1]], %[[S0]] : vector<[8]xf32>
// CHECK: arith.mulf %[[X0]], %[[S1]] : vector<[8]xf32>
