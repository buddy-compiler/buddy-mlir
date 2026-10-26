// RUN: buddy-opt %s -split-input-file -lower-vir-to-vector='vector-width=4' -verify-diagnostics

func.func @dynamic_vl(%a: memref<?xf32>, %vl: index) -> f32 {
  %c0 = arith.constant 0 : index
  %z = arith.constant -0.0 : f32
  // expected-error @below {{a vir.set_vl with results needs a constant vl}}
  %r = vir.set_vl %vl : index {
    %x = vir.load %a[%c0] : memref<?xf32> -> !vir.vec<?xf32>
    %s = vir.reduce %x, %z {kind = "add"} : !vir.vec<?xf32>, f32 -> f32
    vector.yield %s : f32
  } -> f32
  return %r : f32
}

// -----

func.func @from_vector_strip_mined(%v: vector<8xf32>, %o: memref<8xf32>) {
  %c0 = arith.constant 0 : index
  %vl = arith.constant 8 : index
  vir.set_vl %vl : index {
    // expected-error @below {{vir.from_vector needs a vir.set_vl with results}}
    %x = vir.from_vector %v : vector<8xf32> -> !vir.vec<?xf32>
    vir.store %x, %o[%c0] : !vir.vec<?xf32> -> memref<8xf32>
    vector.yield
  }
  return
}

// -----

func.func @from_vector_wrong_length(%v: vector<8xf32>) -> f32 {
  %vl = arith.constant 10 : index
  %z = arith.constant -0.0 : f32
  %r = vir.set_vl %vl : index {
    // expected-error @below {{vir.from_vector needs a fixed vector of vl lanes}}
    %x = vir.from_vector %v : vector<8xf32> -> !vir.vec<?xf32>
    %s = vir.reduce %x, %z {kind = "add"} : !vir.vec<?xf32>, f32 -> f32
    vector.yield %s : f32
  } -> f32
  return %r : f32
}

// -----

// A dynamic vector from above the region: an error whatever the number of
// pieces (here one, vl 4 at width 4).
func.func @outer_vector_one_piece(%x: !vir.vec<?xf32>, %o: memref<4xf32>) -> f32 {
  %c0 = arith.constant 0 : index
  %vl = arith.constant 4 : index
  %z = arith.constant 0.0 : f32
  %r = vir.set_vl %vl : index {
    %y = vir.broadcast %z : f32 -> !vir.vec<?xf32>
    // expected-error @below {{a dynamic vector used in a vir.set_vl with results must be defined in its region}}
    vir.store %x, %o[%c0] : !vir.vec<?xf32> -> memref<4xf32>
    vector.yield %z : f32
  } -> f32
  return %r : f32
}

// -----

// The same with two pieces (vl 8 at width 4).
func.func @outer_vector_two_pieces(%x: !vir.vec<?xf32>, %o: memref<8xf32>) -> f32 {
  %c0 = arith.constant 0 : index
  %vl = arith.constant 8 : index
  %z = arith.constant 0.0 : f32
  %r = vir.set_vl %vl : index {
    %y = vir.broadcast %z : f32 -> !vir.vec<?xf32>
    // expected-error @below {{a dynamic vector used in a vir.set_vl with results must be defined in its region}}
    vir.store %x, %o[%c0] : !vir.vec<?xf32> -> memref<8xf32>
    vector.yield %z : f32
  } -> f32
  return %r : f32
}

// -----

func.func @from_vector_element_type(%v: vector<4xf32>) {
  // expected-error @below {{all of {source, result} have same element type}}
  %x = vir.from_vector %v : vector<4xf32> -> !vir.vec<?xf64>
  return
}

// -----

// The rows of a !vir.vec<2x?xf32> of vl 64 are not one after the other in a
// memref of rows of 128.
func.func @rows_apart(%a: memref<2x128xf32>) -> f32 {
  %vl = arith.constant 64 : index
  %z = arith.constant 0.0 : f32
  %r = vir.set_vl %vl : index {
    // expected-error @below {{the rows of a multi-dimensional dynamic vector must be one after the other in memory}}
    %x = vir.load %a[] : memref<2x128xf32> -> !vir.vec<2x?xf32>
    %s = vir.reduce %x, %z {kind = "add"} : !vir.vec<2x?xf32>, f32 -> f32
    vector.yield %s : f32
  } -> f32
  return %r : f32
}

// -----

func.func @two_layouts(%a: memref<2x64xf32>, %b: memref<64xf32>) -> f32 {
  %vl = arith.constant 64 : index
  %z = arith.constant 0.0 : f32
  %r = vir.set_vl %vl : index {
    %x = vir.load %a[] : memref<2x64xf32> -> !vir.vec<2x?xf32>
    // expected-error @below {{the dynamic vectors of a vir.set_vl with results must have the same static layout and no scaling factor}}
    %y = vir.load %b[] : memref<64xf32> -> !vir.vec<?xf32>
    %s = vir.reduce %x, %z {kind = "add"} : !vir.vec<2x?xf32>, f32 -> f32
    vector.yield %s : f32
  } -> f32
  return %r : f32
}
