// RUN: buddy-opt %s -lower-vir-to-vector='vector-width=64' -canonicalize \
// RUN: | FileCheck %s
// RUN: buddy-opt %s -lower-vir-to-vector='vector-width=16' -canonicalize \
// RUN: | FileCheck %s --check-prefix=NARROW
// RUN: buddy-opt %s \
// RUN:     -lower-vir-to-vector='rvv-vlen=1024 use-scalable=true vector-width=4' \
// RUN: | FileCheck %s --check-prefix=SCALABLE

// The lanes past the main loop of a strip-mined set_vl go to vectors of half
// its width, then a quarter, ..., each at most once and in the order of the
// lanes, while they have at least 512 bits (16 f32), then to the scalar tail
// loop.

// 51 lanes with vectors of 64: no iteration of the main loop, then 32 at 0
// and 16 at 32, then 3 scalars.
// CHECK-LABEL: func.func @short
// CHECK-DAG: %[[C32:.*]] = arith.constant 32 : index
// CHECK:     vector.load %{{.*}}[%{{.*}}] : memref<51xf32>, vector<32xf32>
// CHECK:     vector.load %{{.*}}[%[[C32]]] : memref<51xf32>, vector<16xf32>
// CHECK-NOT: vector<8xf32>
// CHECK:     affine.for %{{.*}} = 48 to 51 {
// CHECK:       memref.load
// With vectors of 16, a half has 8 lanes: no part, 3 vectors then 3 scalars.
// NARROW-LABEL: func.func @short
// NARROW:     affine.for %{{.*}} = 0 to 36 step 16 {
// NARROW-NOT: vector<8xf32>
// NARROW:     affine.for %{{.*}} = 48 to 51 {
func.func @short(%a: memref<51xf32>, %o: memref<51xf32>) {
  %vl = arith.constant 51 : index
  vir.set_vl %vl : index {
    %x = vir.load %a[] : memref<51xf32> -> !vir.vec<?xf32>
    vir.store %x, %o[] : !vir.vec<?xf32> -> memref<51xf32>
    vector.yield
  }
  return
}

// Scalable vectors at VLEN 1024 (vscale 16): vector<[4]xf32> is 64 lanes;
// the rest of 100 lanes (36) goes to a vector<[2]xf32> (32 lanes) and a
// vector<[1]xf32> (16 lanes) when it has as many, tested at run time, then
// to the scalar tail loop.
// SCALABLE-LABEL: func.func @sum
// SCALABLE:     scf.for {{.*}} {
// SCALABLE:       vector.load {{.*}} vector<[4]xf32>
// SCALABLE:     scf.if %{{.*}} {
// SCALABLE:       vector.load {{.*}} vector<[2]xf32>
// SCALABLE:       vector.reduction <add>, %{{.*}}, %{{.*}} : vector<[2]xf32> into f32
// SCALABLE:     scf.if %{{.*}} {
// SCALABLE:       vector.load {{.*}} vector<[1]xf32>
// SCALABLE:     %[[AT:.*]] = arith.select
// SCALABLE:     scf.for %{{.*}} = %[[AT]] to %{{.*}} step %{{.*}} {
// SCALABLE:       memref.load
func.func @sum(%a: memref<100xf32>, %o: memref<1xf32>) {
  %vl = arith.constant 100 : index
  %c0 = arith.constant 0 : index
  vir.set_vl %vl : index {
    %x = vir.load %a[] : memref<100xf32> -> !vir.vec<?xf32>
    %s0 = memref.load %o[%c0] : memref<1xf32>
    %s = vir.reduce %x, %s0 {kind = "add"} : !vir.vec<?xf32>, f32 -> f32
    memref.store %s, %o[%c0] : memref<1xf32>
    vector.yield
  }
  return
}
