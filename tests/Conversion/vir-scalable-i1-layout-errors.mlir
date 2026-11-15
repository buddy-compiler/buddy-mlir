// RUN: buddy-opt %s -lower-vir-to-vector='use-scalable=true rvv-vlen=256' \
// RUN:     -split-input-file -verify-diagnostics

// A scalable access of a memref of i1 reads its bytes from the aligned
// pointer and the strides of the memref: a layout that is not strided has
// none.
#floor = affine_map<(d0) -> (d0 floordiv 4)>
func.func @not_strided(%m: memref<128xi1, #floor>, %o: memref<128xi1>) {
  %vl = arith.constant 128 : index
  vir.set_vl %vl : index {
    // expected-error @+1 {{a scalable access of a memref of i1 needs a strided layout}}
    %k = vir.load %m[] : memref<128xi1, #floor> -> !vir.vec<?xi1>
    vir.store %k, %o[] : !vir.vec<?xi1> -> memref<128xi1>
    vector.yield
  }
  return
}

// -----

// The pointer to the bytes takes the memory space as an LLVM address space,
// which must be an integer.
func.func @symbolic_space(%m: memref<128xi1, "gpu">, %o: memref<128xi1>) {
  %vl = arith.constant 128 : index
  vir.set_vl %vl : index {
    // expected-error @+1 {{a scalable access of a memref of i1 needs an integer memory space}}
    %k = vir.load %m[] : memref<128xi1, "gpu"> -> !vir.vec<?xi1>
    vir.store %k, %o[] : !vir.vec<?xi1> -> memref<128xi1>
    vector.yield
  }
  return
}
