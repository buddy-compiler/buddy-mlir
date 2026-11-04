// RUN: buddy-opt %s -lower-vir-to-vector='use-scalable=true rvv-vlen=128 rvv-vlen-exact=false' \
// RUN: | FileCheck %s --check-prefixes=CHECK,MIN
// RUN: buddy-opt %s -lower-vir-to-vector='rvv-vlen=128 rvv-vlen-exact=false' \
// RUN: | FileCheck %s --check-prefixes=CHECK,FIXED

// With rvv-vlen-exact=false, rvv-vlen is only the minimum VLEN: the code is
// valid on any VLEN of at least rvv-vlen. A strip-mined region steps by
// N * vscale, read at run time: valid on any VLEN, lowered as with the exact
// VLEN.
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
// MIN: %[[VS:.*]] = vector.vscale
// MIN: %[[STEP:.*]] = arith.muli %{{.*}}, %[[VS]] : index
// MIN: scf.for {{.*}} step %[[STEP]] {
// MIN: vector.load {{.*}} : memref<?xf32>, vector<[{{[0-9]+}}]xf32>
// FIXED: vector.load {{.*}} : memref<?xf32>, vector<{{[0-9]+}}xf32>
