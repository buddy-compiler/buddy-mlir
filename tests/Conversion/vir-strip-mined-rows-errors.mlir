// RUN: buddy-opt %s -lower-vir-to-vector="vector-width=4" -verify-diagnostics

// A strip-mined region loops over the rows of one static layout.
func.func @two_layouts(%a: memref<2x8xf32>, %b: memref<4x8xf32>) {
  %vl = arith.constant 8 : index
  vir.set_vl %vl : index {
    %x = vir.load %a[] : memref<2x8xf32> -> !vir.vec<2x?xf32>
    // expected-error @below {{the multi-dimensional dynamic vectors of a vir.set_vl must have the same static layout}}
    %y = vir.load %b[] : memref<4x8xf32> -> !vir.vec<4x?xf32>
    vir.store %x, %a[] : !vir.vec<2x?xf32> -> memref<2x8xf32>
    vector.yield
  }
  return
}
