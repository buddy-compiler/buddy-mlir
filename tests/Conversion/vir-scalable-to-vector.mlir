// RUN: buddy-opt %s -lower-vir-to-vector='vector-width=4 use-scalable=true' \
// RUN: | FileCheck %s
// RUN: buddy-opt %s -lower-vir-to-vector='vector-width=4' \
// RUN: | FileCheck %s --check-prefix=FIXED

// A vector<[4]xf32> has 4 * vscale lanes: the main loop steps by 4 * vscale
// up to vl - vl % (4 * vscale), where the scalar tail loop starts. Both are
// scf.for, since that end is not a valid affine symbol inside other loops.
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

// CHECK-LABEL: func.func @sum
// CHECK: %[[C4:.*]] = arith.constant 4 : index
// CHECK: %[[VSCALE:.*]] = vector.vscale
// CHECK: %[[STEP:.*]] = arith.muli %[[C4]], %[[VSCALE]] : index
// CHECK: %[[REM:.*]] = arith.remsi %{{.*}}, %[[STEP]] : index
// CHECK: %[[END:.*]] = arith.subi %{{.*}}, %[[REM]] : index
// CHECK: scf.for %[[I:.*]] = %{{.*}} to %[[END]] step %[[STEP]] {
// CHECK:   vector.load %{{.*}}[%[[I]]] : memref<100xf32>, vector<[4]xf32>
// CHECK:   vector.reduction <add>, %{{.*}}, %{{.*}} : vector<[4]xf32> into f32
// CHECK: }
// CHECK: %[[C1:.*]] = arith.constant 1 : index
// CHECK: scf.for %{{.*}} = %[[END]] to %{{.*}} step %[[C1]] {
// CHECK:   memref.load
// CHECK:   arith.addf

// Fixed vectors keep the constant step.
// FIXED-LABEL: func.func @sum
// FIXED: affine.for %{{.*}} = #map(%{{.*}}) to #map(%{{.*}}) step 4 {
// FIXED:   vector.load %{{.*}} : memref<100xf32>, vector<4xf32>

// An scf.for in a region without results carries dynamic vectors: in the
// main loop as vectors, in the tail loop as scalars; an unsigned loop stays
// unsigned.
func.func @loop(%a: memref<100xf32>, %lb: i8, %ub: i8, %step: i8) {
  %vl = arith.constant 100 : index
  vir.set_vl %vl : index {
    %x = vir.load %a[] : memref<100xf32> -> !vir.vec<?xf32>
    %r = scf.for unsigned %i = %lb to %ub step %step
        iter_args(%acc = %x) -> (!vir.vec<?xf32>) : i8 {
      %n = arith.addf %acc, %x : !vir.vec<?xf32>
      scf.yield %n : !vir.vec<?xf32>
    }
    vir.store %r, %a[] : !vir.vec<?xf32> -> memref<100xf32>
    vector.yield
  }
  return
}

// CHECK-LABEL: func.func @loop
// CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK:   %[[X:.*]] = vector.load {{.*}} vector<[4]xf32>
// CHECK:   %[[R:.*]] = scf.for unsigned %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ACC:.*]] = %[[X]]) -> (vector<[4]xf32>) : i8 {
// CHECK:     %[[N:.*]] = arith.addf %[[ACC]], %[[X]] : vector<[4]xf32>
// CHECK:     scf.yield %[[N]] : vector<[4]xf32>
// CHECK:   vector.store %[[R]], {{.*}} vector<[4]xf32>
// CHECK: scf.for
// CHECK:   %[[XS:.*]] = memref.load
// CHECK:   %[[RS:.*]] = scf.for unsigned %{{.*}} iter_args(%[[ACCS:.*]] = %[[XS]]) -> (f32) : i8 {
// CHECK:     arith.addf %[[ACCS]], %[[XS]] : f32
// CHECK:   memref.store %[[RS]]

// With a leading static dim, the main and tail loops are inside the loop
// over that dim.
func.func @copy2d(%a: memref<2x100xf32>, %b: memref<2x100xf32>) {
  %vl = arith.constant 100 : index
  vir.set_vl %vl : index {
    %x = vir.load %a[] : memref<2x100xf32> -> !vir.vec<2x?xf32>
    vir.store %x, %b[] : !vir.vec<2x?xf32> -> memref<2x100xf32>
    vector.yield
  }
  return
}

// CHECK-LABEL: func.func @copy2d
// CHECK: affine.for %[[D:.*]] = #map(%{{.*}}) to #map(%{{.*}}) {
// CHECK:   %[[END:.*]] = arith.subi
// CHECK:   scf.for %[[I:.*]] = %{{.*}} to %[[END]] step %{{.*}} {
// CHECK:     vector.load %{{.*}}[%[[D]], %[[I]]] : memref<2x100xf32>, vector<[4]xf32>
// CHECK:   scf.for %[[J:.*]] = %[[END]] to %{{.*}} step %{{.*}} {
// CHECK:     memref.load %{{.*}}[%[[D]], %[[J]]] : memref<2x100xf32>

// An index computed in an scf.for of the region is used through its lowered
// value.
func.func @index_in_loop(%a: memref<8xf32>, %b: memref<8xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  vir.set_vl %c4 : index {
    scf.for %i = %c0 to %c2 step %c1 {
      %off = arith.muli %i, %c4 : index
      %x = vir.load %a[%off] : memref<8xf32> -> !vir.vec<?xf32>
      vir.store %x, %b[%off] : !vir.vec<?xf32> -> memref<8xf32>
    }
    vector.yield
  }
  return
}

// CHECK-LABEL: func.func @index_in_loop
// CHECK: scf.for %[[IV:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK:   scf.for %[[I:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK:     %[[OFF:.*]] = arith.muli %[[I]], %{{.*}} : index
// CHECK:     %[[A:.*]] = arith.addi %[[IV]], %[[OFF]] : index
// CHECK:     vector.load %{{.*}}[%[[A]]] : memref<8xf32>, vector<[4]xf32>
// CHECK:     %[[B:.*]] = arith.addi %[[IV]], %[[OFF]] : index
// CHECK:     vector.store %{{.*}}, %{{.*}}[%[[B]]] : memref<8xf32>, vector<[4]xf32>

// FIXED-LABEL: func.func @index_in_loop
// FIXED: affine.for %[[IV:.*]] = #map(%{{.*}}) to #map(%{{.*}}) step 4 {
// FIXED:   scf.for %[[I:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
// FIXED:     %[[OFF:.*]] = arith.muli %[[I]], %{{.*}} : index
// FIXED:     %[[A:.*]] = arith.addi %[[IV]], %[[OFF]] : index
// FIXED:     vector.load %{{.*}}[%[[A]]] : memref<8xf32>, vector<4xf32>
