// RUN: buddy-opt %s -lower-vir-to-vector='vector-width=8' -canonicalize \
// RUN: | FileCheck %s
// RUN: buddy-opt %s -lower-vir-to-vector='vector-width=64' -canonicalize \
// RUN: | FileCheck %s --check-prefix=WIDE

// The lanes past the main loop of a strip-mined set_vl go to vectors of half
// its width, then a quarter, ..., each at most once and in the order of the
// lanes, before the scalar tail loop: with a constant vl of 15 and vectors of
// 8, one vector of 8, then 4 at 8 and 2 at 12, then one scalar at 14.
// CHECK-LABEL: func.func @add
// CHECK-DAG: %[[C8:.*]] = arith.constant 8 : index
// CHECK-DAG: %[[C12:.*]] = arith.constant 12 : index
// CHECK:     affine.for %[[I:.*]] = 0 to 8 step 8 {
// CHECK:       vector.load %{{.*}}[%[[I]]] : memref<15xf32>, vector<8xf32>
// CHECK:     }
// CHECK:     vector.load %{{.*}}[%[[C8]]] : memref<15xf32>, vector<4xf32>
// CHECK:     vector.store %{{.*}}[%[[C8]]] : memref<15xf32>, vector<4xf32>
// CHECK:     vector.load %{{.*}}[%[[C12]]] : memref<15xf32>, vector<2xf32>
// CHECK:     vector.store %{{.*}}[%[[C12]]] : memref<15xf32>, vector<2xf32>
// CHECK:     affine.for %[[J:.*]] = 14 to 15 {
// CHECK:       memref.load %{{.*}}[%[[J]]] : memref<15xf32>
func.func @add(%a: memref<15xf32>, %b: memref<15xf32>, %o: memref<15xf32>) {
  %vl = arith.constant 15 : index
  vir.set_vl %vl : index {
    %x = vir.load %a[] : memref<15xf32> -> !vir.vec<?xf32>
    %y = vir.load %b[] : memref<15xf32> -> !vir.vec<?xf32>
    %s = arith.addf %x, %y : !vir.vec<?xf32>
    vir.store %s, %o[] : !vir.vec<?xf32> -> memref<15xf32>
    vector.yield
  }
  return
}

// Fewer lanes than the main vector (51 with vectors of 64: no iteration of
// the main loop): 32, then 16, then 2, then one scalar.
// WIDE-LABEL: func.func @short
// WIDE:     vector.load {{.*}} vector<32xf32>
// WIDE:     vector.load {{.*}} vector<16xf32>
// WIDE-NOT: vector<8xf32>
// WIDE-NOT: vector<4xf32>
// WIDE:     vector.load {{.*}} vector<2xf32>
// WIDE:     affine.for %{{.*}} = 50 to 51 {
func.func @short(%a: memref<51xf32>, %o: memref<51xf32>) {
  %vl = arith.constant 51 : index
  vir.set_vl %vl : index {
    %x = vir.load %a[] : memref<51xf32> -> !vir.vec<?xf32>
    vir.store %x, %o[] : !vir.vec<?xf32> -> memref<51xf32>
    vector.yield
  }
  return
}
