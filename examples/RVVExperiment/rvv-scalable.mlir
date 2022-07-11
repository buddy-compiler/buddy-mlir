memref.global "private" @gv : memref<20xf32> = dense<[0. , 1. , 2. , 3. , 4. , 5. , 6. , 7. , 8. , 9. ,
                                                      10., 11., 12., 13., 14., 15., 16., 17., 18., 19.]>

func.func private @printMemrefF32(memref<*xf32>)

func.func @alloc_mem() -> memref<20xf32> {
  %f0 = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %mem = memref.alloc() : memref<20xf32>
  %dim = memref.dim %mem, %c0 : memref<20xf32>
  scf.for %idx = %c0 to %dim step %c1 {
    memref.store %f0, %mem[%idx] : memref<20xf32>
  }
  return %mem : memref<20xf32>
}

func.func @main() -> i32 {
  %mem = memref.get_global @gv : memref<20xf32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  // Load [4]xf32 elements and store into a new array.
  %load_vec1 = vector.load %mem[%c0] : memref<20xf32>, vector<[4]xf32>
  %res1 = call @alloc_mem() : () -> memref<20xf32>
  vector.store %load_vec1, %res1[%c0] : memref<20xf32>, vector<[4]xf32>
  %print_res1 = memref.cast %res1 : memref<20xf32> to memref<*xf32>
  call @printMemrefF32(%print_res1) : (memref<*xf32>) -> ()

  // Load [8]xf32 elements and store into a new array.
  %load_vec2 = vector.load %mem[%c0] : memref<20xf32>, vector<[8]xf32>
  %res2 = call @alloc_mem() : () -> memref<20xf32>
  vector.store %load_vec2, %res2[%c0] : memref<20xf32>, vector<[8]xf32>
  %print_res2 = memref.cast %res2 : memref<20xf32> to memref<*xf32>
  call @printMemrefF32(%print_res2) : (memref<*xf32>) -> ()

  // Load [10]xf32 elements and store into a new array.
  %load_vec3 = vector.load %mem[%c0] : memref<20xf32>, vector<[10]xf32>
  %res3 = call @alloc_mem() : () -> memref<20xf32>
  vector.store %load_vec3, %res3[%c0] : memref<20xf32>, vector<[10]xf32>
  %print_res3 = memref.cast %res3 : memref<20xf32> to memref<*xf32>
  call @printMemrefF32(%print_res3) : (memref<*xf32>) -> ()

  // Configure the register.
  // SEW = 32
  %sew = arith.constant 2 : index
  // LMUL = 2
  %lmul = arith.constant 1 : index
  // AVL = 20
  %avl = arith.constant 20 : index

  // Load vl elements and store into a new array.
  %vl = rvv.setvl %avl, %sew, %lmul : index
  %load_vec4 = rvv.load %mem[%c0], %vl : memref<20xf32>, vector<[4]xf32>, index
  %res4 = call @alloc_mem() : () -> memref<20xf32>
  rvv.store %load_vec4, %res4[%c0], %vl : vector<[4]xf32>, memref<20xf32>, index
  %print_res4 = memref.cast %res4 : memref<20xf32> to memref<*xf32>
  call @printMemrefF32(%print_res4) : (memref<*xf32>) -> ()

  // Load all elements and store into a new array.
  %res5 = call @alloc_mem() : () -> memref<20xf32>
  // While loop.
  %a1, %a2 = scf.while (%curr_avl = %avl, %curr_idx = %c0) : (index, index) -> (index, index) {
    // If avl greater than zero.
    %cond = arith.cmpi sgt, %curr_avl, %c0 : index
    // Pass avl, idx to the after region.
    scf.condition(%cond) %curr_avl, %curr_idx : index, index
  } do {
  ^bb0(%curr_avl : index, %curr_idx : index):
    // Perform the calculation according to the vl.
    %vl5 = rvv.setvl %curr_avl, %sew, %lmul : index
    %load_vec5 = rvv.load %mem[%curr_idx], %vl5 : memref<20xf32>, vector<[4]xf32>, index
    rvv.store %load_vec5, %res5[%curr_idx], %vl5 : vector<[4]xf32>, memref<20xf32>, index
    // Update idx and avl.
    %new_idx = arith.addi %curr_idx, %vl5 : index
    %new_avl = arith.subi %curr_avl, %vl5 : index
    scf.yield %new_avl, %new_idx : index, index
  }
  %print_res5 = memref.cast %res5 : memref<20xf32> to memref<*xf32>
  call @printMemrefF32(%print_res5) : (memref<*xf32>) -> ()

  %ret = arith.constant 0 : i32
  return %ret : i32
}
