#CSR = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ]
}>

// Use linalg.matmul to multiply matrix
func.func @matmul(%A: tensor<4x2xf64, #CSR>,
                  %B: tensor<2x4xf64, #CSR>) -> tensor<4x4xf64, #CSR> {
  %C = bufferization.alloc_tensor() : tensor<4x4xf64, #CSR>
  %D = linalg.matmul
    ins(%A, %B: tensor<4x2xf64, #CSR>, tensor<2x4xf64, #CSR>)
    outs(%C: tensor<4x4xf64, #CSR>) -> tensor<4x4xf64, #CSR>
  return %D: tensor<4x4xf64, #CSR>
}

// @matmul will be rewrited into code like @matmul_expand.
// It use `sparse_tensor.expand` to get the innermost level access pattern of the given tensor.
// This operation is useful to implement calculation kernel on the output tensor.
// The `values` array contains values in the innermost level of the given tensor.
// The `filled` array contains boolean value that indicate whether a coordinate has been filled at the level.
// The `added` array and `count` array are used to store new level-coordinate when the current operating coordinate has never been filled.
// After kernel operation done, the rewrite op will use `sparse_tensor.compress` to compress those access pattern into the output tensor.
func.func @matmul_expand(%A: tensor<4x2xf64, #CSR>, %B: tensor<2x4xf64, #CSR>) -> tensor<4x4xf64, #CSR> {
  %c4 = arith.constant 4 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %false = arith.constant false
  %true = arith.constant true

  %output = bufferization.alloc_tensor() : tensor<4x4xf64, #CSR>
  %ptr_a = sparse_tensor.positions %A {level = 1 : index}
    : tensor<4x2xf64, #CSR> to memref<?xindex>
  %coo_a = sparse_tensor.coordinates %A {level = 1 : index}
    : tensor<4x2xf64, #CSR> to memref<?xindex>
  %val_a = sparse_tensor.values %A
    : tensor<4x2xf64, #CSR> to memref<?xf64>

  %ptr_b = sparse_tensor.positions %B {level = 1 : index}
    : tensor<2x4xf64, #CSR> to memref<?xindex>
  %coo_b = sparse_tensor.coordinates %B {level = 1 : index}
    : tensor<2x4xf64, #CSR> to memref<?xindex>
  %val_b = sparse_tensor.values %B
    : tensor<2x4xf64, #CSR> to memref<?xf64>

  %ret = scf.for %i = %c0 to %c4 step %c1 iter_args(%out = %output) -> (tensor<4x4xf64, #CSR>) {
    %values, %filled, %added, %count = sparse_tensor.expand %output
      : tensor<4x4xf64, #CSR> to memref<?xf64>, memref<?xi1>, memref<?xindex>

    // load slice pointers from A
    // Start bound
    %ptr_as = memref.load %ptr_a[%i] : memref<?xindex>
    %10 = arith.addi %i, %c1 : index
    // End bound
    %ptr_ae = memref.load %ptr_a[%10] : memref<?xindex>

    %final_counter = scf.for %j = %ptr_as to %ptr_ae step %c1 iter_args(%_cnt = %count) -> (index) {
      // load coordinate from A at %j
      %coo_aj = memref.load %coo_a[%j] : memref<?xindex>
      // load value from A at %j
      %val_aj = memref.load %val_a[%j] : memref<?xf64>

      // load pointer range from B
      %ptr_bs = memref.load %ptr_b[%coo_aj] : memref<?xindex>
      %17 = arith.addi %coo_aj, %c1 : index
      %ptr_be = memref.load %ptr_b[%17] : memref<?xindex>

      %added_counter = scf.for %k = %ptr_bs to %ptr_be step %c1 iter_args(%__cnt = %_cnt) -> (index) {
        // load coordinate from B at %k
        %coo_bk = memref.load %coo_b[%k] : memref<?xindex>
        // load values from *return tensor* at %k
        %val_out = memref.load %values[%coo_bk] : memref<?xf64>
        // load values from B at %k
        %val_bk = memref.load %val_b[%k] : memref<?xf64>

        // Multiply value from A and B
        %mul = arith.mulf %val_aj, %val_bk : f64
        // Add the value above to the value existing in output
        %muladd = arith.addf %val_out, %mul : f64

        // Check the value in *return tensor* at %k is filled or not
        %is_filled = memref.load %filled[%coo_bk] : memref<?xi1>
        // True if value is false
        %cmp = arith.cmpi eq, %is_filled, %false : i1
        // If the coodinate there is not filled
        %___cnt = scf.if %cmp -> (index) {
          // set it as filled
          memref.store %true, %filled[%coo_bk] : memref<?xi1>
          // Add the new coordinate into added array
          memref.store %coo_bk, %added[%__cnt] : memref<?xindex>
          // Increase the count
          %28 = arith.addi %__cnt, %c1 : index
          scf.yield %28 : index
        } else {
          // Return the count directly
          scf.yield %__cnt : index
        }
        // Store the final calculated value into output memref
        memref.store %muladd, %values[%coo_bk] : memref<?xf64>
        // return counter
        scf.yield %___cnt : index
      } {"Emitted from" = "linalg.generic"}
      // return counter
      scf.yield %added_counter : index
    } {"Emitted from" = "linalg.generic"}
    // Compress this dimension into the tensor
    %13 = sparse_tensor.compress %values, %filled, %added, %final_counter into %out[%i] : memref<?xf64>, memref<?xi1>, memref<?xindex>, tensor<4x4xf64, #CSR>
    // return the calculated tensor
    scf.yield %13 : tensor<4x4xf64, #CSR>
  } {"Emitted from" = "linalg.generic"}
  %8 = sparse_tensor.load %ret hasInserts : tensor<4x4xf64, #CSR>
  return %8 : tensor<4x4xf64, #CSR>
}
