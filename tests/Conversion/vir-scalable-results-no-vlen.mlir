// RUN: buddy-opt %s -lower-vir-to-vector='use-scalable=true vector-width=4' \
// RUN:   -verify-diagnostics

// The scalable vectors of a vir.set_vl with results are sized for the VLEN
// of the target: without rvv-vlen, vscale is not known.
func.func @dot(%a: memref<128xf32>) -> f32 {
  %vl = arith.constant 128 : index
  %z = arith.constant 0.0 : f32
  // expected-error @below {{a vir.set_vl with results on scalable vectors needs rvv-vlen}}
  %r = vir.set_vl %vl : index {
    %x = vir.load %a[] : memref<128xf32> -> !vir.vec<?xf32>
    %s = vir.reduce %x, %z {kind = "add"} : !vir.vec<?xf32>, f32 -> f32
    vector.yield %s : f32
  } -> f32
  return %r : f32
}
