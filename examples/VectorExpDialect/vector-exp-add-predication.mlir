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
  %dim = memref.dim %input1, %c0 : memref<?xi32>

  // Configure the register.
  // SEW = 32
  %sew = arith.constant 2 : index
  // LMUL = 2
  %lmul = arith.constant 1 : index

  // Constant mask configuration.
  %mask = arith.constant dense<1> : vector<[4]xi1>

  // While loop for strip-mining.
  %tmpAVL, %tmpIdx = scf.while (%avl = %dim, %idx = %c0) : (index, index) -> (index, index) {
    // If avl greater than zero.
    %cond = arith.cmpi sgt, %avl, %c0 : index
    // Pass avl, idx to the after region.
    scf.condition(%cond) %avl, %idx : index, index
  } do {
  ^bb0(%avl : index, %idx : index):
    // Perform the calculation according to the vl.
    %vl = rvv.setvl %avl, %sew, %lmul : index
    %vl_i32 = arith.index_cast %vl : index to i32
    %vec_input1 = vector_exp.predication %mask, %vl_i32 : vector<[4]xi1>, i32 {
      %ele = vector.load %input1[%idx] : memref<?xi32>, vector<[4]xi32>
      vector.yield %ele : vector<[4]xi32>
    } : vector<[4]xi32>
    %vec_input2 = vector_exp.predication %mask, %vl_i32 : vector<[4]xi1>, i32 {
      %ele = vector.load %input2[%idx] : memref<?xi32>, vector<[4]xi32>
      vector.yield %ele : vector<[4]xi32>
    } : vector<[4]xi32>
    %result_vector = rvv.add %vec_input1, %vec_input2, %vl : vector<[4]xi32>, vector<[4]xi32>, index
    vector_exp.predication %mask, %vl_i32 : vector<[4]xi1>, i32 {
      vector.store %result_vector, %output[%idx] : memref<?xi32>, vector<[4]xi32>
      vector.yield
    } : () -> ()
    // Update idx and avl.
    %new_idx = arith.addi %idx, %vl : index
    %new_avl = arith.subi %avl, %vl : index
    scf.yield %new_avl, %new_idx : index, index
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
