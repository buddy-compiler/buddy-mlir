// RUN: buddy-opt %s -lower-vir-to-vector='target-vector-bits=512 target-vector-regs=32' | FileCheck %s --check-prefix=X512
// RUN: buddy-opt %s -lower-vir-to-vector='target-vector-bits=256 target-vector-regs=16' | FileCheck %s --check-prefix=X256
// RUN: buddy-opt %s -lower-vir-to-vector='target-vector-bits=128 target-vector-regs=32' | FileCheck %s --check-prefix=ARM128
// RUN: not buddy-opt %s -lower-vir-to-vector='target-vector-bits=512' 2>&1 | FileCheck %s --check-prefix=PAIR
// RUN: not buddy-opt %s -lower-vir-to-vector='target-vector-bits=0 target-vector-regs=32' 2>&1 | FileCheck %s --check-prefix=PAIR
// RUN: not buddy-opt %s -lower-vir-to-vector='target-vector-bits=192 target-vector-regs=32' 2>&1 | FileCheck %s --check-prefix=PAIR
// RUN: not buddy-opt %s -lower-vir-to-vector='target-vector-bits=512 target-vector-regs=-1' 2>&1 | FileCheck %s --check-prefix=PAIR
// RUN: not buddy-opt %s -lower-vir-to-vector='target-vector-bits=512 target-vector-regs=32 rvv-vlen=256' 2>&1 | FileCheck %s --check-prefix=CONFLICT
// RUN: not buddy-opt %s -lower-vir-to-vector='target-vector-bits=512 target-vector-regs=32 use-scalable=true' 2>&1 | FileCheck %s --check-prefix=CONFLICT
// RUN: not buddy-opt %s -lower-vir-to-vector='target-vector-bits=512 target-vector-regs=32 use-scalable=false' 2>&1 | FileCheck %s --check-prefix=CONFLICT

// PAIR: target-vector-bits must be a power of two >= 8 and target-vector-regs must be positive; specify both
// CONFLICT: target-vector-bits/regs cannot be combined with rvv-vlen or use-scalable

// Five live vectors give 32/5 or 16/5 registers per vector, rounded
// down to a power of two. Widths are independent of the test host.
func.func @pressure(%a: memref<?xf32>, %b: memref<?xf32>,
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
// X512-LABEL: func.func @pressure
// X512: vector.load {{.*}} vector<64xf32>
// X512-NOT: vector<[
// X256-LABEL: func.func @pressure
// X256: vector.load {{.*}} vector<16xf32>
// X256-NOT: vector<[
// ARM128-LABEL: func.func @pressure
// ARM128: vector.load {{.*}} vector<16xf32>
// ARM128-NOT: vector<[

// Dynamic loop bounds can come from non-affine operations.
func.func @dynamic_n(%a: memref<?xf32>, %n: index, %m: index) {
  %bound = arith.minsi %n, %m : index
  vir.set_vl %bound : index {
    %x = vir.load %a[] : memref<?xf32> -> !vir.vec<?xf32>
    vir.store %x, %a[] : !vir.vec<?xf32> -> memref<?xf32>
    vector.yield
  }
  return
}
// X512-LABEL: func.func @dynamic_n
// X512: arith.minsi
// X512: scf.for
// X512: vector.load
// X512: scf.for
// X512: memref.load
// X512-NOT: affine.for
// X256-LABEL: func.func @dynamic_n
// X256: scf.for
// X256: vector.load
// X256: scf.for
// ARM128-LABEL: func.func @dynamic_n
// ARM128: scf.for
// ARM128: vector.load
// ARM128: scf.for
