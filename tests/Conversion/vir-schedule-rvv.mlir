// RUN: buddy-opt %s -vir-schedule='verbose=true' \
// RUN: | FileCheck %s --check-prefix=GENERIC
// RUN: buddy-opt %s -vir-schedule='rvv=true verbose=true' \
// RUN: | FileCheck %s --check-prefix=RVV
// RUN: buddy-opt %s -vir-schedule='rvv=true' \
// RUN:     -lower-vir-to-vector='rvv-vlen=256 verbose=true' \
// RUN: | FileCheck %s --check-prefix=REQ

// With rvv, -vir-schedule counts the vector registers as -lower-vir-to-vector
// does with an RVV VLEN: a broadcast that is only the scalar operand of
// arithmetic (here of vfmul.vf) stays in a scalar register and is not
// counted.
//
// Counted (any target), %s, %a, %b and %p are live at the mulf: 4; with %p
// before %b is loaded, 3. On RVV, %s is not counted: 3 either way, and the
// original order is kept, of the Req_Group -lower-vir-to-vector finds.
func.func @scalar_operand(%m: memref<2x256xf32>, %o: memref<256xf32>,
                          %f: f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %vl = arith.constant 256 : index
  vir.set_vl %vl : index {
    %s = vir.broadcast %f : f32 -> !vir.vec<?xf32>
    %a = vir.load %m[%c0, %c0] : memref<2x256xf32> -> !vir.vec<?xf32>
    %b = vir.load %m[%c1, %c0] : memref<2x256xf32> -> !vir.vec<?xf32>
    %p = arith.mulf %a, %s : !vir.vec<?xf32>
    %q = arith.addf %p, %b : !vir.vec<?xf32>
    vir.store %q, %o[%c0] : !vir.vec<?xf32> -> memref<256xf32>
    vector.yield
  }
  return
}

// GENERIC: vir-schedule: MaxLive 4 -> 3 in the block of vir.set_vl
// GENERIC-LABEL: func.func @scalar_operand
// GENERIC: vir.set_vl
// GENERIC-NOT: vir.load %{{.*}}[%c1
// GENERIC: arith.mulf
// GENERIC-NEXT: vir.load %{{.*}}[%c1

// RVV: vir-schedule: MaxLive 3 -> 3, original order kept in the block of
// RVV-LABEL: func.func @scalar_operand
// RVV: vir.set_vl
// RVV-NEXT: vir.broadcast
// RVV-NEXT: vir.load %{{.*}}[%c0
// RVV-NEXT: vir.load %{{.*}}[%c1
// RVV-NEXT: arith.mulf

// REQ: Final Max Active Vectors (Req_Group): 3
