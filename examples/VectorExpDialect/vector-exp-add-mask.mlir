module {
  func.func private @printMemrefI32(memref<*xi32>)

  func.func @alloc_filled_input(%size: index) -> memref<?xi32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %mem = memref.alloc(%size) : memref<?xi32>
    scf.for %idx = %c0 to %size step %c1 {
      %val = arith.index_cast %idx : index to i32
      memref.store %val, %mem[%idx] : memref<?xi32>
    }
    return %mem : memref<?xi32>
  }

  func.func @alloc_filled_output(%size: index) -> memref<?xi32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0 : i32
    %mem = memref.alloc(%size) : memref<?xi32>
    scf.for %idx = %c0 to %size step %c1 {
      memref.store %cst, %mem[%idx] : memref<?xi32>
    }
    return %mem : memref<?xi32>
  }

func.func @vector_add(%input1: memref<?xi32>, %input2: memref<?xi32>, %output: memref<?xi32>) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %cst = arith.constant 0 : i32
  %pass_thr = vector.splat %cst : vector<4xi32>
  %dim = memref.dim %input1, %c0 : memref<?xi32>
  %iter = arith.divui %dim, %c4 : index
  %rem = arith.remui %dim, %c4 : index
  %tail = arith.cmpi ne, %rem, %c0 : index
  affine.for %idx = %c0 to %iter {
    %cur_idx = arith.muli %idx, %c4 : index
    %vec_input1 = affine.vector_load %input1[%idx * 4] : memref<?xi32>, vector<4xi32>
    %vec_input2 = affine.vector_load %input2[%idx * 4] : memref<?xi32>, vector<4xi32>
    %vec_output = arith.addi %vec_input1, %vec_input2 : vector<4xi32>
    affine.vector_store %vec_output, %output[%idx * 4] : memref<?xi32>, vector<4xi32>
  }
  // Tail processing
  scf.if %tail {
    %cur_idx = arith.muli %iter, %c4 : index
    %mask = vector.create_mask %rem : vector<4xi1>
    %vec_input1 = vector.maskedload %input1[%cur_idx], %mask, %pass_thr : memref<?xi32>, vector<4xi1>, vector<4xi32> into vector<4xi32>
    %vec_input2 = vector.maskedload %input2[%cur_idx], %mask, %pass_thr : memref<?xi32>, vector<4xi1>, vector<4xi32> into vector<4xi32>
    %vec_output = arith.addi %vec_input1, %vec_input2 : vector<4xi32>
    vector.maskedstore %output[%cur_idx], %mask, %vec_output : memref<?xi32>, vector<4xi1>, vector<4xi32>
  }
  return
}

  func.func @main() -> i32 {
    %c10 = arith.constant 10 : index

    %input1 = call @alloc_filled_input(%c10) : (index) -> memref<?xi32>
    %input2 = call @alloc_filled_input(%c10) : (index) -> memref<?xi32>
    %output = call @alloc_filled_output(%c10) : (index) -> memref<?xi32>

    call @vector_add(%input1, %input2, %output) : (memref<?xi32>, memref<?xi32>, memref<?xi32>) -> ()

    // Print output.
    %print_output = memref.cast %output : memref<?xi32> to memref<*xi32>
    call @printMemrefI32(%print_output) : (memref<*xi32>) -> ()

    memref.dealloc %input1 : memref<?xi32>
    memref.dealloc %input2 : memref<?xi32>
    memref.dealloc %output : memref<?xi32>

    %ret = arith.constant 0 : i32
    return %ret : i32
  }
}
