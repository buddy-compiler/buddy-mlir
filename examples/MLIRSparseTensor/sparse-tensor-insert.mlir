#SV = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed" ]
}>

func.func @main() {
  %s0 = bufferization.alloc_tensor() : tensor<1024xf64, #SV>

  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %v1 = arith.constant 1.1 : f64

  scf.for %i = %c0 to %c4 step %c1 iter_args(%v = %v1) -> f64 {
    sparse_tensor.insert %v into %s0[%i] : tensor<1024xf64, #SV>
    %next_v = arith.addf %v, %v1 : f64
    scf.yield %next_v : f64
  }

  // Notify the sparse compiler that %s0 has been inserted and sparse storage need to be finalized.
  %s1 = sparse_tensor.load %s0 hasInserts : tensor<1024xf64, #SV>

  sparse_tensor.foreach in %s1 : tensor<1024xf64, #SV> do {
    ^bb0(%i: index, %v: f64):
      vector.print %i : index
      vector.print %v : f64
  }

  return
}
