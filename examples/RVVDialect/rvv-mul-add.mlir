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

  // Configure the register.
  // SEW = 32
  %sew = arith.constant 2 : index
  // LMUL = 2
  %lmul = arith.constant 1 : index
  // AVL = 8
  %avl8 = arith.constant 8 : index
  // Load vl elements.
  %vl8 = rvv.setvl %avl8, %sew, %lmul : index

  %vec_c0_i32 = rvv.load %mem_i32[%c0], %vl8 : memref<20xi32>, vector<[8]xi32>, index
  %vec_c10_i32 = rvv.load %mem_i32[%c10], %vl8 : memref<20xi32>, vector<[8]xi32>, index
  %res_mem = call @alloc_mem_i32() : () -> memref<20xi32>
  
  %res_mul = rvv.mul %vec_c0_i32, %vec_c10_i32, %vl8 : vector<[8]xi32>, vector<[8]xi32>, index
  %res_add = rvv.add %res_mul, %vec_c0_i32, %vl8 : vector<[8]xi32>, vector<[8]xi32>, index

  rvv.store %res_add, %res_mem[%c0], %vl8 : vector<[8]xi32>, memref<20xi32>, index

  %print_res = memref.cast %res_mem : memref<20xi32> to memref<*xi32>
  call @printMemrefI32(%print_res) : (memref<*xi32>) -> ()

  %ret = arith.constant 0 : i32
  return %ret : i32
}
