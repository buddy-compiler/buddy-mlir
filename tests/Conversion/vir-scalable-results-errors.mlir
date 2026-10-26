// RUN: buddy-opt %s -split-input-file \
// RUN:   -lower-vir-to-vector='use-scalable=true rvv-vlen=256' -verify-diagnostics

// What a vir.set_vl with results does not lower to scalable vectors yet.

func.func @from_vector(%v: vector<128xf32>) -> f32 {
  %vl = arith.constant 128 : index
  %z = arith.constant 0.0 : f32
  %r = vir.set_vl %vl : index {
    // expected-error @below {{vir.from_vector needs fixed vectors}}
    %x = vir.from_vector %v : vector<128xf32> -> !vir.vec<?xf32>
    %s = vir.reduce %x, %z {kind = "add"} : !vir.vec<?xf32>, f32 -> f32
    vector.yield %s : f32
  } -> f32
  return %r : f32
}

// -----

func.func @strided(%a: memref<128xf32, strided<[2]>>) -> f32 {
  %vl = arith.constant 128 : index
  %z = arith.constant 0.0 : f32
  %r = vir.set_vl %vl : index {
    // expected-error @below {{a vir.set_vl with results on scalable vectors needs unit-stride accesses}}
    %x = vir.load %a[] : memref<128xf32, strided<[2]>> -> !vir.vec<?xf32>
    %s = vir.reduce %x, %z {kind = "add"} : !vir.vec<?xf32>, f32 -> f32
    vector.yield %s : f32
  } -> f32
  return %r : f32
}

// -----

func.func @i1(%a: memref<128xf32>, %b: memref<128xi1>) -> f32 {
  %vl = arith.constant 128 : index
  %z = arith.constant 0.0 : f32
  %r = vir.set_vl %vl : index {
    %x = vir.load %a[] : memref<128xf32> -> !vir.vec<?xf32>
    %zv = vir.broadcast %z : f32 -> !vir.vec<?xf32>
    %m = arith.cmpf ogt, %x, %zv : !vir.vec<?xf32>
    // expected-error @below {{a memref of i1 needs fixed vectors}}
    vir.store %m, %b[] : !vir.vec<?xi1> -> memref<128xi1>
    %s = vir.reduce %x, %z {kind = "add"} : !vir.vec<?xf32>, f32 -> f32
    vector.yield %s : f32
  } -> f32
  return %r : f32
}

// -----

// 128 f32 lanes at VLEN 256: 16 registers.
func.func @shuffle_too_long(%q: memref<2x64xf32>) -> f32 {
  %vl = arith.constant 64 : index
  %z = arith.constant 0.0 : f32
  %r = vir.set_vl %vl : index {
    %x = vir.load %q[] : memref<2x64xf32> -> !vir.vec<2x?xf32>
    // expected-error @below {{vir.shuffle needs a dynamic vector of 128 lanes in one register group of at most 8}}
    %y = vir.shuffle %x, %x [1, 0] : !vir.vec<2x?xf32>
    %s = vir.reduce %y, %z {kind = "add"} : !vir.vec<2x?xf32>, f32 -> f32
    vector.yield %s : f32
  } -> f32
  return %r : f32
}

// -----

// The size is checked with the element type of the shuffle (f32), not the
// anchor's (f16): vector<[32]xf32> is 16 registers.
func.func @shuffle_extf(%a: memref<2x64xf16>) -> f32 {
  %vl = arith.constant 64 : index
  %z = arith.constant 0.0 : f32
  %r = vir.set_vl %vl : index {
    %h = vir.load %a[] : memref<2x64xf16> -> !vir.vec<2x?xf16>
    %x = vir.extf %h : !vir.vec<2x?xf16> -> !vir.vec<2x?xf32>
    // expected-error @below {{vir.shuffle needs a dynamic vector of 128 lanes in one register group of at most 8}}
    %y = vir.shuffle %x, %x [1, 0] : !vir.vec<2x?xf32>
    %s = vir.reduce %y, %z {kind = "add"} : !vir.vec<2x?xf32>, f32 -> f32
    vector.yield %s : f32
  } -> f32
  return %r : f32
}
