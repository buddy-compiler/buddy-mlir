func.func @conv1d_linalg(%in : memref<?xf32>, %filter : memref<?xf32>, %out : memref<?xf32>) -> () {
  linalg.conv_1d ins(%in, %filter : memref<?xf32>, memref<?xf32>)
                outs(%out : memref<?xf32>)
  return
}

// Produces the same result as the above function, but with the dap fir opeartion.
func.func @conv1d_buddy(%in : memref<?xf32>, %filter : memref<?xf32>, %out : memref<?xf32>) -> () {
  dap.fir %in, %filter, %out : memref<?xf32>, memref<?xf32>, memref<?xf32>
  return
}
