// RUN: buddy-opt %s -lower-vir-to-vector='use-scalable=true' \
// RUN:   -verify-diagnostics

// For any RVV VLEN (use-scalable without rvv-vlen), the scalable pieces of
// a vir.set_vl with results would have more lanes than the ones they are
// cut for on a larger VLEN (vector<[N]xT> is N * VLEN / 64 lanes): rejected.
func.func @dot(%a: memref<64xf32>) -> f32 {
  %vl = arith.constant 64 : index
  %z = arith.constant 0.0 : f32
  // expected-error @below {{a vir.set_vl with results on scalable vectors needs rvv-vlen, the exact VLEN}}
  %r = vir.set_vl %vl : index {
    %x = vir.load %a[] : memref<64xf32> -> !vir.vec<?xf32>
    %s = vir.reduce %x, %z {kind = "add"} : !vir.vec<?xf32>, f32 -> f32
    vector.yield %s : f32
  } -> f32
  return %r : f32
}
