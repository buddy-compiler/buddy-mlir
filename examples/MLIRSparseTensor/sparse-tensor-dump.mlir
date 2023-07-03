#SparseMatrix = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed", "compressed" ]
}>

#SparseVector = #sparse_tensor.encoding<{
  dimLevelType = ["compressed"]
}>

#CSR = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ]
}>

func.func private @printMemrefInd(%arg0 : memref<*xindex>)

func.func @dump_vector(%arg: tensor<1024xf32, #SparseVector>) {
  %c0 = arith.constant 0 : index
  %pad = arith.constant -1 : index
  %vpad = arith.constant -1.0 : f32

  // Print pointers
  %p = sparse_tensor.positions %arg { level = 0 : index }
        : tensor<1024xf32, #SparseVector> to memref<?xindex>
  %p1 = vector.transfer_read %p[%c0], %pad : memref<?xindex>, vector<2xindex>
  vector.print %p1 : vector<2xindex>

  // Print Indices(Coordinate)
  %c = sparse_tensor.coordinates %arg { level = 0 : index }
        : tensor<1024xf32, #SparseVector> to memref<?xindex>
  %c1 = vector.transfer_read %c[%c0], %pad : memref<?xindex>, vector<4xindex>
  vector.print %c1 : vector<4xindex>

  // Print values (non-zero entries)
  %v = sparse_tensor.values %arg : tensor<1024xf32, #SparseVector> to memref<?xf32>
  %v1 = vector.transfer_read %v[%c0], %vpad : memref<?xf32>, vector<4xf32>
  vector.print %v1 : vector<4xf32>

  return
}

func.func @dump_matrix(%arg: tensor<1024x1024xf32, #SparseMatrix>) {
  %c0 = arith.constant 0 : index
  %pad = arith.constant -1 : index
  %vpad = arith.constant -1.0 : f32

  vector.print %c0 : index
  // Print pointers
  %p0 = sparse_tensor.positions %arg { level = 0 : index }
        : tensor<1024x1024xf32, #SparseMatrix> to memref<?xindex>
  %vp0 = vector.transfer_read %p0[%c0], %pad : memref<?xindex>, vector<2xindex>
  vector.print %vp0 : vector<2xindex>

  // Print Indices(Coordinate)
  %coo0 = sparse_tensor.coordinates %arg { level = 0 : index }
        : tensor<1024x1024xf32, #SparseMatrix> to memref<?xindex>
  %vcoo0 = vector.transfer_read %coo0[%c0], %pad : memref<?xindex>, vector<5xindex>
  vector.print %vcoo0 : vector<5xindex>

  // Print dimension 1
  %c1 = arith.constant 1 : index
  vector.print %c1 : index
  %p1 = sparse_tensor.positions %arg { level = 1 : index }
        : tensor<1024x1024xf32, #SparseMatrix> to memref<?xindex>
  %vp1 = vector.transfer_read %p1[%c0], %pad : memref<?xindex>, vector<6xindex>
  vector.print %vp1 : vector<6xindex>
  %coo1 = sparse_tensor.coordinates %arg { level = 1 : index }
        : tensor<1024x1024xf32, #SparseMatrix> to memref<?xindex>
  %vcoo1 = vector.transfer_read %coo1[%c0], %pad : memref<?xindex>, vector<6xindex>
  vector.print %vcoo1 : vector<6xindex>

  %v = sparse_tensor.values %arg : tensor<1024x1024xf32, #SparseMatrix> to memref<?xf32>
  %v0 = vector.transfer_read %v[%c0], %vpad : memref<?xf32>, vector<6xf32>
  vector.print %v0 : vector<6xf32>

  return
}

func.func @dump_csr(%arg: tensor<1024x1024xf32, #CSR>) {
  %c0 = arith.constant 0 : index
  %pad = arith.constant 99 : index
  %vpad = arith.constant -1.0 : f32

  // Print pointers
  %p0 = sparse_tensor.positions %arg { level = 1 : index }
        : tensor<1024x1024xf32, #CSR> to memref<?xindex>
  %vp0 = vector.transfer_read %p0[%c0], %pad : memref<?xindex>, vector<1025xindex>
  vector.print %vp0 : vector<1025xindex>

  // Print Indices(Coordinate)
  %coo0 = sparse_tensor.coordinates %arg { level = 1 : index }
        : tensor<1024x1024xf32, #CSR> to memref<?xindex>
  %vcoo0 = vector.transfer_read %coo0[%c0], %pad : memref<?xindex>, vector<6xindex>
  vector.print %vcoo0 : vector<6xindex>

  // Print values
  %v = sparse_tensor.values %arg : tensor<1024x1024xf32, #CSR> to memref<?xf32>
  %v0 = vector.transfer_read %v[%c0], %vpad : memref<?xf32>, vector<6xf32>
  vector.print %v0 : vector<6xf32>

  return
}

func.func @main() {
  // Create a Sparse Vector.
  // The first row is a list of coordinate of the non-zero entries.
  // The second row contains actual value of those non-zero entries.
  %m = arith.constant sparse<
    [[0, 1], [0, 4], [4, 8], [32, 64], [128, 256], [512, 1023]],
    [ 1.1, 2.2, 3.3, 4.4, 4.4, 5.5 ]
  > : tensor<1024x1024xf32>
  // Convert into Sparse Storage
  // pointers: [
  //   [0, 5],
  //   [0, 2, 3, 4, 5, 6],
  // ]
  // indices: [
  //   [0, 4, 32, 128, 512],
  //   [1, 4, 8, 64, 256, 1023],
  // ]
  // values: [ 1.1, 2.2, 3.3, 4.4, 4.4, 5.5 ]
  %sm = sparse_tensor.convert %m : tensor<1024x1024xf32> to tensor<1024x1024xf32, #SparseMatrix>
  call @dump_matrix(%sm) : (tensor<1024x1024xf32, #SparseMatrix>) -> ()

  // Convert the sparse matrix into CSR format
  // The first dimension is store in dense.
  // The second dimension:
  // pointers[1]: [0, 2, 2, 2, 2, 3, 3, 3, 3, 4, ..., 5, ..., 6] # length = 1025
  // indices[1]: [1, 4, 8, 64, 256, 1023]
  // values: [ 1.1, 2.2, 3.3, 4.4, 4.4, 5.5 ]
  //
  // let s = pointers[1][i], e = pointers[1][i+1],
  // range [ s, e ) is the slice access of the indices and values for row `i`.
  %csrm = sparse_tensor.convert %m : tensor<1024x1024xf32> to tensor<1024x1024xf32, #CSR>
  call @dump_csr(%csrm) : (tensor<1024x1024xf32, #CSR>) -> ()

  // Create a Sparse vector [0, 1.1, ..., 2.2, ..., 3.3, ..., 4.4, 0]
  %v = arith.constant sparse<
    [[1], [8], [37], [1023]],
    [1.1, 2.2, 3.3, 4.4]
  > : tensor<1024xf32>
  // Convert into sparse storage
  // pointers[0]: [ 0, 4 ]
  // indices[0]: [ 1, 8, 37, 1023 ]
  // values: [ 1.1, 2.2, 3.3, 4.4 ]
  //
  // Content of pointers[0][0] and pointers[0][1] denote that elements appear at
  // range [0, 4) in the indices and values array.
  // Indices[0][..] and values[..] contains the coordinates and value of non-zero elements.
  %sv = sparse_tensor.convert %v : tensor<1024xf32> to tensor<1024xf32, #SparseVector>
  call @dump_vector(%sv) : (tensor<1024xf32, #SparseVector>) -> ()

  return
}
