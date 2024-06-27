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

func.func @matmul(%a : memref<10x10xi32>, %b : memref<10x10xi32>, %c : memref<10x10xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  %aRow = memref.dim %a, %c0 : memref<10x10xi32>
  %aCol = memref.dim %a, %c1 : memref<10x10xi32>
  %bRow = memref.dim %b, %c0 : memref<10x10xi32>
  %bCol = memref.dim %b, %c1 : memref<10x10xi32>

  // Configure the register.
  // SEW = 32
  %sew = arith.constant 2 : index
  // LMUL = 2
  %lmul = arith.constant 1 : index

  affine.for %idx0 = 0 to %bRow {
    affine.for %idx1 = 0 to %aRow {
      %aEle = affine.load %a[%idx1, %idx0] : memref<10x10xi32>
      // While loop for strip-mining.
      %tmpAVL, %tmpIdx = scf.while (%avl = %bCol, %idx = %c0) : (index, index) -> (index, index) {
        // If avl greater than zero.
        %cond = arith.cmpi sgt, %avl, %c0 : index
        // Pass avl, idx to the after region.
        scf.condition(%cond) %avl, %idx : index, index
      } do {
      ^bb0(%avl : index, %idx : index):
        // Perform the calculation according to the vl.
        %vl = rvv.setvl %avl, %sew, %lmul : index
        %vl_i32 = arith.index_cast %vl : index to i32
        %mask = vector.create_mask %vl : vector<[4]xi1>
        %input_vector = vector_exp.predication %mask, %vl_i32 : vector<[4]xi1>, i32 {
          %ele = vector.load %b[%idx0, %idx] : memref<10x10xi32>, vector<[4]xi32>
          vector.yield %ele : vector<[4]xi32>
        } : vector<[4]xi32>
        %mul_vector = rvv.mul %input_vector, %aEle, %vl : vector<[4]xi32>, i32, index
        %c_vector = vector_exp.predication %mask, %vl_i32 : vector<[4]xi1>, i32 {
          %ele = vector.load %c[%idx1, %idx] : memref<10x10xi32>, vector<[4]xi32>
          vector.yield %ele : vector<[4]xi32>
        } : vector<[4]xi32>
        %result_vector = rvv.add %mul_vector, %c_vector, %vl : vector<[4]xi32>, vector<[4]xi32>, index
        vector_exp.predication %mask, %vl_i32 : vector<[4]xi1>, i32 {
          vector.store %result_vector, %c[%idx1, %idx] : memref<10x10xi32>, vector<[4]xi32>
          vector.yield
        } : () -> ()
        // Update idx and avl.
        %new_idx = arith.addi %idx, %vl : index
        %new_avl = arith.subi %avl, %vl : index
        scf.yield %new_avl, %new_idx : index, index
      }
    }
  }
  return
}

func.func @main() -> i32 {
  %mem_i32 = memref.get_global @gv_i32 : memref<10x10xi32>
  %result_mem = call @alloc_mem_i32() : () -> memref<10x10xi32>

  call @matmul(%mem_i32, %mem_i32, %result_mem) : (memref<10x10xi32>, memref<10x10xi32>, memref<10x10xi32>) -> ()

  %print_result_mem = memref.cast %result_mem : memref<10x10xi32> to memref<*xi32>
  call @printMemrefI32(%print_result_mem) : (memref<*xi32>) -> ()

  %ret = arith.constant 0 : i32
  return %ret : i32
}
