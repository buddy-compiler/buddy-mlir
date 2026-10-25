// RUN: buddy-opt %s -split-input-file \
// RUN:     -lower-vir-to-vector='rvv-vlen=256 verbose=true' | FileCheck %s

// The register pressure of a region (Req_Group) counts the dynamic vectors
// live at every op, those in nested loops included, and gives the vector
// size: on RVV, LMUL = min(8, 2^floor(log2(32 / Req_Group))) registers of
// VLEN / 32 f32 lanes.

// In the loop: the four q, captured, and the four accumulators, carried,
// are live with k and the result of the first fma: 10, LMUL 2, 16 lanes.
// The initial value and the loop results are not counted in the loop.
func.func @loop_carried(%q: memref<4x256xf32>, %k: memref<64x256xf32>,
                        %o: memref<4x256xf32>, %n: index) -> f32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %vl = arith.constant 256 : index
  %zero = arith.constant 0.0 : f32
  %r0 = vir.set_vl %vl : index {
    %q0 = vir.load %q[%c0, %c0] : memref<4x256xf32> -> !vir.vec<?xf32>
    %q1 = vir.load %q[%c1, %c0] : memref<4x256xf32> -> !vir.vec<?xf32>
    %q2 = vir.load %q[%c2, %c0] : memref<4x256xf32> -> !vir.vec<?xf32>
    %q3 = vir.load %q[%c3, %c0] : memref<4x256xf32> -> !vir.vec<?xf32>
    %init = vir.constant {value = 0.0 : f32} : !vir.vec<?xf32>
    %r:4 = scf.for %i = %c0 to %n step %c1
        iter_args(%a0 = %init, %a1 = %init, %a2 = %init, %a3 = %init)
        -> (!vir.vec<?xf32>, !vir.vec<?xf32>, !vir.vec<?xf32>, !vir.vec<?xf32>) {
      %kv = vir.load %k[%i, %c0] : memref<64x256xf32> -> !vir.vec<?xf32>
      %b0 = vir.fma %q0, %kv, %a0 : !vir.vec<?xf32>
      %b1 = vir.fma %q1, %kv, %a1 : !vir.vec<?xf32>
      %b2 = vir.fma %q2, %kv, %a2 : !vir.vec<?xf32>
      %b3 = vir.fma %q3, %kv, %a3 : !vir.vec<?xf32>
      scf.yield %b0, %b1, %b2, %b3
          : !vir.vec<?xf32>, !vir.vec<?xf32>, !vir.vec<?xf32>, !vir.vec<?xf32>
    }
    vir.store %r#0, %o[%c0, %c0] : !vir.vec<?xf32> -> memref<4x256xf32>
    vir.store %r#1, %o[%c1, %c0] : !vir.vec<?xf32> -> memref<4x256xf32>
    vir.store %r#2, %o[%c2, %c0] : !vir.vec<?xf32> -> memref<4x256xf32>
    vir.store %r#3, %o[%c3, %c0] : !vir.vec<?xf32> -> memref<4x256xf32>
    %s = vir.reduce %r#0, %zero {kind = "add"} : !vir.vec<?xf32>, f32 -> f32
    vector.yield %s : f32
  } -> f32
  return %r0 : f32
}

// CHECK: 10 live at %{{.*}} = vir.fma
// CHECK: Final Max Active Vectors (Req_Group): 10
// CHECK: Calculated LMUL = 2, Size_vec = (2 * 256) / 32 = 16
// CHECK-LABEL: func.func @loop_carried
// CHECK: vector.fma {{.*}} : vector<16xf32>

// -----

// Vectors used after the loop and not in it are live in the loop too: 11
// at the add in the loop (x0 and x1, which it uses, x2 .. x9, used after
// it, and its result), 9 at most after the loop; LMUL 2.
func.func @live_across(%m: memref<10x256xf32>, %o: memref<256xf32>,
                       %n: index) -> f32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c5 = arith.constant 5 : index
  %c6 = arith.constant 6 : index
  %c7 = arith.constant 7 : index
  %c8 = arith.constant 8 : index
  %c9 = arith.constant 9 : index
  %vl = arith.constant 256 : index
  %zero = arith.constant 0.0 : f32
  %r = vir.set_vl %vl : index {
    %x0 = vir.load %m[%c0, %c0] : memref<10x256xf32> -> !vir.vec<?xf32>
    %x1 = vir.load %m[%c1, %c0] : memref<10x256xf32> -> !vir.vec<?xf32>
    %x2 = vir.load %m[%c2, %c0] : memref<10x256xf32> -> !vir.vec<?xf32>
    %x3 = vir.load %m[%c3, %c0] : memref<10x256xf32> -> !vir.vec<?xf32>
    %x4 = vir.load %m[%c4, %c0] : memref<10x256xf32> -> !vir.vec<?xf32>
    %x5 = vir.load %m[%c5, %c0] : memref<10x256xf32> -> !vir.vec<?xf32>
    %x6 = vir.load %m[%c6, %c0] : memref<10x256xf32> -> !vir.vec<?xf32>
    %x7 = vir.load %m[%c7, %c0] : memref<10x256xf32> -> !vir.vec<?xf32>
    %x8 = vir.load %m[%c8, %c0] : memref<10x256xf32> -> !vir.vec<?xf32>
    %x9 = vir.load %m[%c9, %c0] : memref<10x256xf32> -> !vir.vec<?xf32>
    scf.for %i = %c0 to %n step %c1 {
      %s = arith.addf %x0, %x1 : !vir.vec<?xf32>
      vir.store %s, %o[%c0] : !vir.vec<?xf32> -> memref<256xf32>
    }
    %t3 = arith.addf %x2, %x3 : !vir.vec<?xf32>
    %t4 = arith.addf %t3, %x4 : !vir.vec<?xf32>
    %t5 = arith.addf %t4, %x5 : !vir.vec<?xf32>
    %t6 = arith.addf %t5, %x6 : !vir.vec<?xf32>
    %t7 = arith.addf %t6, %x7 : !vir.vec<?xf32>
    %t8 = arith.addf %t7, %x8 : !vir.vec<?xf32>
    %t9 = arith.addf %t8, %x9 : !vir.vec<?xf32>
    %s = vir.reduce %t9, %zero {kind = "add"} : !vir.vec<?xf32>, f32 -> f32
    vector.yield %s : f32
  } -> f32
  return %r : f32
}

// CHECK: 11 live at %{{.*}} = arith.addf
// CHECK: 9 live at %{{.*}} = arith.addf
// CHECK: Final Max Active Vectors (Req_Group): 11

// -----

// On RVV, a broadcast that only feeds the scalar operand of arithmetic
// (vfmul.vf) is not counted: x, its square and the product, 3, LMUL 8.
// The vector size is no wider than the constant vl: 64 lanes, not 256.
func.func @broadcast_and_vl(%a: memref<64xf32>, %f: f32) {
  %vl = arith.constant 64 : index
  vir.set_vl %vl : index {
    %x = vir.load %a[] : memref<64xf32> -> !vir.vec<?xf32>
    %b = vir.broadcast %f : f32 -> !vir.vec<?xf32>
    %y = arith.mulf %x, %x : !vir.vec<?xf32>
    %z = arith.mulf %y, %b : !vir.vec<?xf32>
    %w = arith.addf %z, %x : !vir.vec<?xf32>
    vir.store %w, %a[] : !vir.vec<?xf32> -> memref<64xf32>
    vector.yield
  }
  return
}

// CHECK: Final Max Active Vectors (Req_Group): 3
// CHECK: Calculated LMUL = 8, Size_vec = (8 * 256) / 32 = 64
// CHECK-LABEL: func.func @broadcast_and_vl
// CHECK: arith.mulf {{.*}} : vector<64xf32>
