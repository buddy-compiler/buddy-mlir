// RUN: buddy-opt %s -lower-vir-to-vector='vector-width=16' -verify-diagnostics

// An op the lowering of a strip-mined set_vl does not lower fails the pass
// instead of being dropped (with its effect).
func.func @sin(%a: memref<51xf32>, %o: memref<51xf32>) {
  %vl = arith.constant 51 : index
  vir.set_vl %vl : index {
    %x = vir.load %a[] : memref<51xf32> -> !vir.vec<?xf32>
    // expected-error @below {{unsupported operation in a strip-mined vir.set_vl: math.sin}}
    %y = math.sin %x : !vir.vec<?xf32>
    vir.store %y, %o[] : !vir.vec<?xf32> -> memref<51xf32>
    vector.yield
  }
  return
}
