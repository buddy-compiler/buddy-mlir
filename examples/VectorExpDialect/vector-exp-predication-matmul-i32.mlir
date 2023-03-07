memref.global "private" @gv_i32 : memref<10x10xi32> = dense<[[0 , 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9],
                                                             [0 , 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9],
                                                             [0 , 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9],
                                                             [0 , 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9],
                                                             [0 , 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9],
                                                             [0 , 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9],
                                                             [0 , 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9],
                                                             [0 , 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9],
                                                             [0 , 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9],
                                                             [0 , 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9]]>

func.func private @printMemrefI32(memref<*xi32>)

func.func @alloc_mem_i32() -> memref<10x10xi32> {
  %i0 = arith.constant 0 : i32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %mem = memref.alloc() : memref<10x10xi32>
  scf.for %idx0 = %c0 to %c10 step %c1 {
    scf.for %idx1 = %c0 to %c10 step %c1 {
      memref.store %i0, %mem[%idx0, %idx1] : memref<10x10xi32>
    }
  }
  return %mem : memref<10x10xi32>
}

func.func @main() -> i32 {
  %mem_i32 = memref.get_global @gv_i32 : memref<10x10xi32>
  %result_mem = call @alloc_mem_i32() : () -> memref<10x10xi32>

  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  %c1 = arith.constant 1 : index

  %aRow = memref.dim %mem_i32, %c0 : memref<10x10xi32>
  %aCol = memref.dim %mem_i32, %c1 : memref<10x10xi32>
  %bRow = memref.dim %mem_i32, %c0 : memref<10x10xi32>
  %bCol = memref.dim %mem_i32, %c1 : memref<10x10xi32>
  %bCol_i32 = arith.index_cast %bCol : index to i32

  // Configure the register.
  // SEW = 32
  %sew = arith.constant 2 : i32
  // LMUL = 2
  %lmul = arith.constant 1 : i32

  affine.for %idx0 = 0 to %bRow {
    affine.for %idx1 = 0 to %aRow {
      %aEle = affine.load %mem_i32[%idx1, %idx0] : memref<10x10xi32>
      // While loop for strip-mining.
      %tmpAVL, %tmpIdx = scf.while (%avl = %bCol_i32, %idx = %c0) : (i32, index) -> (i32, index) {
        // If avl greater than zero.
        %cond = arith.cmpi sgt, %avl, %c0_i32 : i32
        // Pass avl, idx to the after region.
        scf.condition(%cond) %avl, %idx : i32, index
      } do {
      ^bb0(%avl : i32, %idx : index):
        // Perform the calculation according to the vl.
        %vl = rvv.setvl %avl, %sew, %lmul : i32
        %vl_ind = arith.index_cast %vl : i32 to index
        %mask = vector.create_mask %vl_ind : vector<[4]xi1>
        %input_vector = vector_exp.predication %mask, %vl : vector<[4]xi1>, i32 {
          %ele = vector.load %mem_i32[%idx0, %idx] : memref<10x10xi32>, vector<[4]xi32>
          vector.yield %ele : vector<[4]xi32>
        } : vector<[4]xi32>
        %mul_vector = rvv.mul %input_vector, %aEle, %vl : vector<[4]xi32>, i32, i32
        %c_vector = vector_exp.predication %mask, %vl : vector<[4]xi1>, i32 {
          %ele = vector.load %result_mem[%idx1, %idx] : memref<10x10xi32>, vector<[4]xi32>
          vector.yield %ele : vector<[4]xi32>
        } : vector<[4]xi32>
        %result_vector = rvv.add %mul_vector, %c_vector, %vl : vector<[4]xi32>, vector<[4]xi32>, i32
        vector_exp.predication %mask, %vl : vector<[4]xi1>, i32 {
          vector.store %result_vector, %result_mem[%idx1, %idx] : memref<10x10xi32>, vector<[4]xi32>
          vector.yield
        } : () -> ()
        // Update idx and avl.
        %new_idx = arith.addi %idx, %vl_ind : index
        %new_avl = arith.subi %avl, %vl : i32
        scf.yield %new_avl, %new_idx : i32, index
      }
    }
  }

  %print_result_mem = memref.cast %result_mem : memref<10x10xi32> to memref<*xi32>
  call @printMemrefI32(%print_result_mem) : (memref<*xi32>) -> ()

  %ret = arith.constant 0 : i32
  return %ret : i32
}
