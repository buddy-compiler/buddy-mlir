#SortedCOO = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed-nu", "singleton" ]
}>

//func.func private @printMemrefF64()

/// Operating on sparse matrix
///
/// [
///  [0.0, 0.0, 1.1, 0.0],
///  [2.2, 0.0, 0.0, 0.0],
///  [0.0, 0.0, 0.0, 0.0],
///  [0.0, 3.3, 0.0, 0.0],
/// ]
///
func.func @main() {
  // Pack the coordinate list into a sparse tensor
  // <https://en.wikipedia.org/wiki/Sparse_matrix#Coordinate_list_(COO)>
  %data = arith.constant dense<[1.1, 2.2, 3.3]> : tensor<3xf64>
  %idxs = arith.constant dense<[[0,2], [1,0], [3,1]]> : tensor<3x2xindex>
  %sm = sparse_tensor.pack %data, %idxs : tensor<3xf64>, tensor<3x2xindex> to tensor<4x4xf64, #SortedCOO>

  // Iterate all the non-zero entry and print their coordinate and value
  // Expect output:
  // row col val
  // 0   2   1   1   0   2  3   1   3
  sparse_tensor.foreach in %sm : tensor<4x4xf64, #SortedCOO> do {
    ^bb0(%row:index, %col: index, %val: f64) :
      vector.print %row : index
      vector.print %col : index
      vector.print %val : f64
  }

  // Unpack the packed sparse tensor back to values, coordinate tensor, plus the amount of the non-zero.
  %val, %coords, %count = sparse_tensor.unpack %sm : tensor<4x4xf64, #SortedCOO>
      to tensor<3xf64>, tensor<3x2xindex>, i32

  %c0 = arith.constant 0 : index
  %f0 = arith.constant 0.0 : f64

  // Print the value of the sparse tensor
  // Expect ( 1.1, 2.2, 3.3 )
  %vval = vector.transfer_read %val[%c0], %f0 : tensor<3xf64>, vector<3xf64>
  vector.print %vval : vector<3xf64>

  // print the coordinate of the non-zero
  // Expect ( (0, 2), ( 1, 0 ), ( 3, 1 ) )
  %vcoords = vector.transfer_read %coords[%c0, %c0], %c0 : tensor<3x2xindex>, vector<3x2xindex>
  vector.print %vcoords : vector<3x2xindex>

  // Print the amount of the sparse tensor
  // Expect 3
  vector.print %count : i32

  return
}
