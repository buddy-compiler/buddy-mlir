// RUN: buddy-opt %s -lower-vir-to-vector='use-scalable=true rvv-vlen=256 vector-width=64' \
// RUN:   -split-input-file -verify-diagnostics

// An explicit vector-width is checked too: vector<[64]xf32> is 32 registers.
func.func @shuffle_wide(%q: memref<2x64xf32>) -> f32 {
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

// index elements are checked as 64-bit integers: vector<[64]xi64>.
func.func @shuffle_index_wide(%a: memref<2x64xindex>) -> i32 {
  %vl = arith.constant 64 : index
  %zero = arith.constant 0 : i32
  %r = vir.set_vl %vl : index {
    %x = vir.load %a[] : memref<2x64xindex> -> !vir.vec<2x?xindex>
    // expected-error @below {{vir.shuffle needs a dynamic vector of 128 lanes in one register group of at most 8}}
    %y = vir.shuffle %x, %x [1, 0] : !vir.vec<2x?xindex>
    vector.yield %zero : i32
  } -> i32
  return %r : i32
}
