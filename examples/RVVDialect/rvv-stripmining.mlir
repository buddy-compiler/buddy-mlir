memref.global "private" @gv_i32 : memref<20xi32> = dense<[0 , 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9 ,
                                                          10, 11, 12, 13, 14, 15, 16, 17, 18, 19]>

func.func private @printMemrefI32(memref<*xi32>)

func.func @alloc_mem_i32() -> memref<20xi32> {
  %i0 = arith.constant 0 : i32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %mem = memref.alloc() : memref<20xi32>
  %dim = memref.dim %mem, %c0 : memref<20xi32>
  scf.for %idx = %c0 to %dim step %c1 {
    memref.store %i0, %mem[%idx] : memref<20xi32>
  }
  return %mem : memref<20xi32>
}

func.func @main() -> i32 {
  %mem_i32 = memref.get_global @gv_i32 : memref<20xi32>
  %c0 = arith.constant 0 : index
  %c10 = arith.constant 10 : index
  %i2 = arith.constant 2 : i32

  // Configure the register.
  // SEW = 32
  %sew = arith.constant 2 : index
  // LMUL = 2
  %lmul = arith.constant 1 : index

  %init_avl = memref.dim %mem_i32, %c0 : memref<20xi32>
  %init_idx = arith.constant 0 : index
  %res = memref.alloc() : memref<20xi32>

  // While loop for strip-mining.
  %a1, %a2 = scf.while (%avl = %init_avl, %idx = %init_idx) : (index, index) -> (index, index) {
    // If avl greater than zero.
    %cond = arith.cmpi sgt, %avl, %c0 : index
    // Pass avl, idx to the after region.
    scf.condition(%cond) %avl, %idx : index, index
  } do {
  ^bb0(%avl : index, %idx : index):
    // Perform the calculation according to the vl.
    %vl = rvv.setvl %avl, %sew, %lmul : index
    %input_vector = rvv.load %mem_i32[%idx], %vl : memref<20xi32>, vector<[8]xi32>, index
    %result_vector = rvv.add %input_vector, %i2, %vl : vector<[8]xi32>, i32, index
    rvv.store %result_vector, %res[%idx], %vl : vector<[8]xi32>, memref<20xi32>, index
    // Update idx and avl.
    %new_idx = arith.addi %idx, %vl : index
    %new_avl = arith.subi %avl, %vl : index
    scf.yield %new_avl, %new_idx : index, index
  }
  %result = vector.load %res[%c0] : memref<20xi32>, vector<20xi32>
  vector.print %result : vector<20xi32>

  %ret = arith.constant 0 : i32
  return %ret : i32
}
