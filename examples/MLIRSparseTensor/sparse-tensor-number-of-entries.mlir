// This is an example of operation `sparse_tensor.number_of_entries` in MLIR SparseTensor Dialect.
// The `sparse_tensor.number_of_entries` operation accpet a tensor with sparse attribute, then return the number
// of the non-zero entry in the given sparse tensor.

#SparseMatrix = #sparse_tensor.encoding<{
  dimLevelType = ["compressed", "compressed"]
}>

func.func @main() {
  %d0 = arith.constant dense<[
      [1.0, 0.0, 0.0, 0.0],
      [0.0, 0.0, 2.0, 0.0],
      [0.0, 0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0, 3.0]
    ]> : tensor<4x4xf64>

  %s0 = sparse_tensor.convert %d0 : tensor<4x4xf64> to tensor<4x4xf64, #SparseMatrix>
  %n0 = sparse_tensor.number_of_entries %s0 : tensor<4x4xf64, #SparseMatrix>

  // This should print 3
  vector.print %n0 : index

  return
}
