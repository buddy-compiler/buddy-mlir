// RUN: buddy-opt %s -lower-vir-to-vector='use-scalable=true rvv-vlen=256' \
// RUN: | FileCheck %s --check-prefixes=CHECK,V256
// RUN: buddy-opt %s -lower-vir-to-vector='use-scalable=true rvv-vlen=1024' \
// RUN: | FileCheck %s --check-prefixes=CHECK,V1024
// RUN: not buddy-opt %s -lower-vir-to-vector='use-scalable=true' 2>&1 \
// RUN: | FileCheck %s --check-prefix=ERR
// RUN: not buddy-opt %s -lower-vir-to-vector='use-scalable=true rvv-vlen=32' 2>&1 \
// RUN: | FileCheck %s --check-prefix=ERR32

// On RVV, vscale = VLEN / 64: vector<[N]xT> is a group of LMUL = N * SEW / 64
// registers whatever the VLEN, so the scalable vector size only depends on
// the register pressure. rvv-vlen is the minimum VLEN, for the clamp to a
// constant vl.

// ERR: use-scalable without vector-width needs rvv-vlen
// vscale = VLEN / 64 must be at least 1.
// ERR32: rvv-vlen must be at least 64 with use-scalable, got 32

// One vector live: LMUL 8.
func.func @low_pressure(%a: memref<?xf32>, %n: index) {
  %one = arith.constant 1.0 : f32
  vir.set_vl %n : index {
    %x = vir.load %a[] : memref<?xf32> -> !vir.vec<?xf32>
    %b = vir.broadcast %one : f32 -> !vir.vec<?xf32>
    %y = arith.addf %x, %b : !vir.vec<?xf32>
    vir.store %y, %a[] : !vir.vec<?xf32> -> memref<?xf32>
    vector.yield
  }
  return
}

// CHECK-LABEL: func.func @low_pressure
// CHECK: vector.vscale
// CHECK: vector.load {{.*}} vector<[16]xf32>
// CHECK: vector.store {{.*}} vector<[16]xf32>

// Five vectors live: Size_Group = 32 / 5 = 6, LMUL 4.
func.func @high_pressure(%a: memref<?xf32>, %b: memref<?xf32>,
                         %c: memref<?xf32>, %d: memref<?xf32>,
                         %e: memref<?xf32>, %n: index) {
  vir.set_vl %n : index {
    %va = vir.load %a[] : memref<?xf32> -> !vir.vec<?xf32>
    %vb = vir.load %b[] : memref<?xf32> -> !vir.vec<?xf32>
    %vc = vir.load %c[] : memref<?xf32> -> !vir.vec<?xf32>
    %vd = vir.load %d[] : memref<?xf32> -> !vir.vec<?xf32>
    %ve = vir.load %e[] : memref<?xf32> -> !vir.vec<?xf32>
    %s0 = arith.addf %va, %vb : !vir.vec<?xf32>
    %s1 = arith.addf %s0, %vc : !vir.vec<?xf32>
    %s2 = arith.addf %s1, %vd : !vir.vec<?xf32>
    %s3 = arith.addf %s2, %ve : !vir.vec<?xf32>
    vir.store %s3, %a[] : !vir.vec<?xf32> -> memref<?xf32>
    vector.yield
  }
  return
}

// CHECK-LABEL: func.func @high_pressure
// CHECK: vector.load {{.*}} vector<[8]xf32>

// A constant vl of 100 holds 64 lanes at LMUL 8 and VLEN 256, but only
// 2 * 1024 / 32 = 64 at LMUL 2 and VLEN 1024.
func.func @clamped(%a: memref<100xf32>) {
  %vl = arith.constant 100 : index
  vir.set_vl %vl : index {
    %x = vir.load %a[] : memref<100xf32> -> !vir.vec<?xf32>
    %y = arith.addf %x, %x : !vir.vec<?xf32>
    vir.store %y, %a[] : !vir.vec<?xf32> -> memref<100xf32>
    vector.yield
  }
  return
}

// CHECK-LABEL: func.func @clamped
// V256: vector.load {{.*}} vector<[16]xf32>
// V1024: vector.load {{.*}} vector<[4]xf32>

// The smallest scalable vector, [1]: at VLEN 1024, 16 f32 lanes
// (LMUL 1/2), more than the constant vl of 8.
func.func @smallest(%a: memref<8xf32>) {
  %vl = arith.constant 8 : index
  vir.set_vl %vl : index {
    %x = vir.load %a[] : memref<8xf32> -> !vir.vec<?xf32>
    %y = arith.addf %x, %x : !vir.vec<?xf32>
    vir.store %y, %a[] : !vir.vec<?xf32> -> memref<8xf32>
    vector.yield
  }
  return
}

// CHECK-LABEL: func.func @smallest
// V256: vector.load {{.*}} vector<[2]xf32>
// V1024: vector.load {{.*}} vector<[1]xf32>
