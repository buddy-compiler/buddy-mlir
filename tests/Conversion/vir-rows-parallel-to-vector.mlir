// RUN: buddy-opt %s -lower-vir-to-vector="vector-width=4" | FileCheck %s

// The rows of a vir.set_vl without results (the leading static dims of its
// dynamic vectors) run in an scf.parallel only when each row writes elements
// of its own and reads no element that another row writes; else in serial
// affine.for loops.

// Each row reads %a and writes its own row of %o, which is allocated, so it
// does not overlap %a.
func.func @parallel(%a: memref<2x8xf32>) -> memref<2x8xf32> {
  %vl = arith.constant 8 : index
  %o = memref.alloc() : memref<2x8xf32>
  vir.set_vl %vl : index {
    %x = vir.load %a[] : memref<2x8xf32> -> !vir.vec<2x?xf32>
    %y = vir.load %o[] : memref<2x8xf32> -> !vir.vec<2x?xf32>
    %z = arith.addf %x, %y : !vir.vec<2x?xf32>
    vir.store %z, %o[] : !vir.vec<2x?xf32> -> memref<2x8xf32>
    vector.yield
  }
  return %o : memref<2x8xf32>
}

// CHECK-LABEL: func.func @parallel
// CHECK: scf.parallel
// CHECK-NOT: affine.for %{{.*}} = #map(%{{.*}}) to #map(%c2{{.*}})

// Each row reads a global and writes its own row of %o, allocated.
memref.global "private" constant @g : memref<2x8xf32> = dense<1.0>

func.func @global() -> memref<2x8xf32> {
  %vl = arith.constant 8 : index
  %a = memref.get_global @g : memref<2x8xf32>
  %o = memref.alloc() : memref<2x8xf32>
  vir.set_vl %vl : index {
    %x = vir.load %a[] : memref<2x8xf32> -> !vir.vec<2x?xf32>
    vir.store %x, %o[] : !vir.vec<2x?xf32> -> memref<2x8xf32>
    vector.yield
  }
  return %o : memref<2x8xf32>
}

// CHECK-LABEL: func.func @global
// CHECK: scf.parallel

// A scalar read and written once per row, at the same address in every row.
func.func @shared_scalar(%a: memref<2x8xf32>, %s: memref<f32>)
    -> memref<2x8xf32> {
  %vl = arith.constant 8 : index
  %one = arith.constant 1.0 : f32
  %o = memref.alloc() : memref<2x8xf32>
  %t = memref.alloc() : memref<f32>
  vir.set_vl %vl : index {
    %x = vir.load %a[] : memref<2x8xf32> -> !vir.vec<2x?xf32>
    vir.store %x, %o[] : !vir.vec<2x?xf32> -> memref<2x8xf32>
    %n = memref.load %t[] : memref<f32>
    %m = arith.addf %n, %one : f32
    memref.store %m, %t[] : memref<f32>
    vector.yield
  }
  return %o : memref<2x8xf32>
}

// CHECK-LABEL: func.func @shared_scalar
// CHECK-NOT: scf.parallel
// CHECK: affine.for %{{.*}} = #map(%{{.*}}) to #map(%c2{{.*}})
// CHECK: memref.store
// CHECK-LABEL: func.func @overlapping

// A memref whose rows are at the same addresses (stride 0).
func.func @overlapping(%a: memref<2x8xf32>) -> memref<8xf32> {
  %vl = arith.constant 8 : index
  %o = memref.alloc() : memref<8xf32>
  %v = memref.reinterpret_cast %o to offset: [0], sizes: [2, 8],
      strides: [0, 1] : memref<8xf32> to memref<2x8xf32, strided<[0, 1]>>
  vir.set_vl %vl : index {
    %x = vir.load %a[] : memref<2x8xf32> -> !vir.vec<2x?xf32>
    vir.store %x, %v[] : !vir.vec<2x?xf32> -> memref<2x8xf32, strided<[0, 1]>>
    vector.yield
  }
  return %o : memref<8xf32>
}

// CHECK-NOT: scf.parallel
// CHECK: affine.for %{{.*}} = #map(%{{.*}}) to #map(%c2{{.*}})
// CHECK-LABEL: func.func @other_row

// Each row reads the next row of the memref it writes.
func.func @other_row(%a: memref<2x8xf32>) -> memref<3x8xf32> {
  %vl = arith.constant 8 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %o = memref.alloc() : memref<3x8xf32>
  vir.set_vl %vl : index {
    %x = vir.load %o[%c1, %c0] : memref<3x8xf32> -> !vir.vec<2x?xf32>
    vir.store %x, %o[%c0, %c0] : !vir.vec<2x?xf32> -> memref<3x8xf32>
    vector.yield
  }
  return %o : memref<3x8xf32>
}

// CHECK-NOT: scf.parallel
// CHECK: affine.for %{{.*}} = #map(%{{.*}}) to #map(%c2{{.*}})
// CHECK-LABEL: func.func @call_alias

// Each row reads the next row of the result of a call, which may be the
// allocated %o.
func.func private @identity(%m: memref<3x8xf32>) -> memref<3x8xf32>

func.func @call_alias() -> memref<3x8xf32> {
  %vl = arith.constant 8 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %o = memref.alloc() : memref<3x8xf32>
  %v = call @identity(%o) : (memref<3x8xf32>) -> memref<3x8xf32>
  vir.set_vl %vl : index {
    %x = vir.load %v[%c1, %c0] : memref<3x8xf32> -> !vir.vec<2x?xf32>
    vir.store %x, %o[%c0, %c0] : !vir.vec<2x?xf32> -> memref<3x8xf32>
    vector.yield
  }
  return %o : memref<3x8xf32>
}

// CHECK-NOT: scf.parallel
// CHECK: affine.for %{{.*}} = #map(%{{.*}}) to #map(%c2{{.*}})
