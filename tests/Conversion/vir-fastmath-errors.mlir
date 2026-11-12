// RUN: buddy-opt %s -lower-vir-to-vector='use-scalable=true rvv-vlen=256' \
// RUN:     -verify-diagnostics

// An unsupported op with fast-math flags is reported, as without flags: its
// flags have no lowered op to go to.
func.func @unsupported_fastmath(%a: memref<16xf32>, %out: memref<16xf32>) {
  %vl = arith.constant 16 : index
  vir.set_vl %vl : index {
    %x = vir.load %a[] : memref<16xf32> -> !vir.vec<?xf32>
    // expected-error @+1 {{unsupported operation in a strip-mined vir.set_vl: math.sin}}
    %y = math.sin %x fastmath<nnan> : !vir.vec<?xf32>
    vir.store %y, %out[] : !vir.vec<?xf32> -> memref<16xf32>
    vector.yield
  }
  return
}
