// RUN: buddy-opt %s -split-input-file -lower-vir-to-vector="vector-width=4" -verify-diagnostics

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

// -----

// A dynamic vector from outside the region has no vector of the main loop
// nor scalar of the tail loop (nor rows looped over).
func.func @outer_vector(%x: !vir.vec<2x?xf32>, %o: memref<2x8xf32>) {
  %vl = arith.constant 8 : index
  %z = arith.constant 0.0 : f32
  vir.set_vl %vl : index {
    %dummy = vir.broadcast %z : f32 -> !vir.vec<?xf32>
    // expected-error @below {{a dynamic vector used in a vir.set_vl must be defined in its region}}
    vir.store %x, %o[] : !vir.vec<2x?xf32> -> memref<2x8xf32>
    vector.yield
  }
  return
}

// -----

func.func @outer_vector_1d(%x: !vir.vec<?xf32>, %a: memref<8xf32>) {
  %vl = arith.constant 8 : index
  vir.set_vl %vl : index {
    %y = vir.load %a[] : memref<8xf32> -> !vir.vec<?xf32>
    // expected-error @below {{a dynamic vector used in a vir.set_vl must be defined in its region}}
    %s = arith.addf %x, %y : !vir.vec<?xf32>
    vir.store %s, %a[] : !vir.vec<?xf32> -> memref<8xf32>
    vector.yield
  }
  return
}
