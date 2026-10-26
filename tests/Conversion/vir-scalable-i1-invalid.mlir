// RUN: buddy-opt %s -lower-vir-to-vector='use-scalable=true rvv-vlen=256' \
// RUN:   -split-input-file -verify-diagnostics
// RUN: buddy-opt %s -lower-vir-to-vector='vector-width=4 use-scalable=true' \
// RUN:   -split-input-file -verify-diagnostics

// A memref of i1 holds a byte per element, a vector of i1 a bit per lane:
// scalable vectors cannot access it yet (fixed vectors scalarize the
// accesses).

func.func @copy_bool(%a: memref<128xi1>, %b: memref<128xi1>) {
  %vl = arith.constant 128 : index
  vir.set_vl %vl : index {
    // expected-error @+1 {{a memref of i1 needs fixed vectors}}
    %x = vir.load %a[] : memref<128xi1> -> !vir.vec<?xi1>
    vir.store %x, %b[] : !vir.vec<?xi1> -> memref<128xi1>
    vector.yield
  }
  return
}

// -----

// The i1 of a comparison of f32 stored to a memref of i1.
func.func @store_compare(%a: memref<128xf32>, %b: memref<128xi1>) {
  %vl = arith.constant 128 : index
  %zero = arith.constant 0.0 : f32
  vir.set_vl %vl : index {
    %x = vir.load %a[] : memref<128xf32> -> !vir.vec<?xf32>
    %z = vir.broadcast %zero : f32 -> !vir.vec<?xf32>
    %m = arith.cmpf ogt, %x, %z : !vir.vec<?xf32>
    // expected-error @+1 {{a memref of i1 needs fixed vectors}}
    vir.store %m, %b[] : !vir.vec<?xi1> -> memref<128xi1>
    vector.yield
  }
  return
}
