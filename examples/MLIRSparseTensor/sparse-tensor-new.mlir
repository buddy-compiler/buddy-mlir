!Filename = !llvm.ptr<i8>

#SparseMatrix = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed", "compressed" ]
}>

// Get tensor filename from env $TENSOR+index
func.func private @getTensorFilename(index) -> (!Filename)
func.func private @printMemrefI32(memref<*xi32>) -> ()

//
// Computes C = A x B with all matrices dense.
//
func.func @matmul( %A: tensor<4x8xi32, #SparseMatrix>, %B: tensor<8x4xi32, #SparseMatrix>) -> tensor<4x4xi32, #SparseMatrix> {
  %C = bufferization.alloc_tensor() : tensor<4x4xi32, #SparseMatrix>
  %D = linalg.matmul
    ins(%A, %B: tensor<4x8xi32, #SparseMatrix>, tensor<8x4xi32, #SparseMatrix>)
    outs(%C: tensor<4x4xi32, #SparseMatrix>) -> tensor<4x4xi32, #SparseMatrix>
  return %D: tensor<4x4xi32, #SparseMatrix>
}

func.func @main() {
  // Get tensor filename from env variable TENSOR0
  %c0 = arith.constant 0 : index
  %file0 = call @getTensorFilename(%c0) : (index) -> (!Filename)
  // Get tensor filename from env variable TENSOR1
  %c1 = arith.constant 1 : index
  %file1 = call @getTensorFilename(%c1) : (index) -> (!Filename)

  // Read tensor from file and convert them to tensor<m x n x i32, #attr>
  %sa = sparse_tensor.new %file0 : !Filename to tensor<4x8xi32, #SparseMatrix>
  %sb = sparse_tensor.new %file1 : !Filename to tensor<8x4xi32, #SparseMatrix>

  // Call the kernel
  %1 = call @matmul(%sa, %sb)
       : (tensor<4x8xi32, #SparseMatrix>, tensor<8x4xi32, #SparseMatrix>)
          -> tensor<4x4xi32, #SparseMatrix>

  // Extract numerical values to an array
  %val1 = sparse_tensor.values %1 : tensor<4x4xi32, #SparseMatrix> to memref<?xi32>
  // Cast the array to a printable unrank memref
  %val_buf_1 = memref.cast %val1 : memref<?xi32> to memref<*xi32>
  call @printMemrefI32(%val_buf_1) : (memref<*xi32>) -> ()

  %c2 = arith.constant 2 : index
  %file2 = call @getTensorFilename(%c2) : (index) -> (!Filename)
  sparse_tensor.out %1, %file2 : tensor<4x4xi32, #SparseMatrix>, !Filename

  return
}
