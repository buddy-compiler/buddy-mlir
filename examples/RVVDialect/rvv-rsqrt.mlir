memref.global "private" @gv_f32 : memref<20xf32> = dense<[0. , 1. , 2. , 3. , 4. , 5. , 6. , 7. , 8. , 9. ,
                                                          10., 11., 12., 13., 14., 15., 16., 17., 18., 19.]>

func.func private @printMemrefF32(memref<*xf32>)

func.func @alloc_mem_f32() -> memref<20xf32> {
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
  %mem_f32 = memref.get_global @gv_f32 : memref<20xf32>
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

  %load_vec_f32 = rvv.load %mem_f32[%c0], %vl8 : memref<20xf32>, vector<[8]xf32>, index
  %res_rsqrt_mem = call @alloc_mem_f32() : () -> memref<20xf32>

  // Test rsqrt in RVV Dialect.
  %res_rvv = rvv.rsqrt %load_vec_f32, %vl8 : vector<[8]xf32>, index
  rvv.store %res_rvv, %res_rsqrt_mem[%c0], %vl8 : vector<[8]xf32>, memref<20xf32>, index
  // Test rsqrt in Math Dialect.
  %res_math = math.rsqrt %load_vec_f32 : vector<[8]xf32>
  rvv.store %res_math, %res_rsqrt_mem[%c10], %vl8 : vector<[8]xf32>, memref<20xf32>, index

  %print_res_rsqrt= memref.cast %res_rsqrt_mem : memref<20xf32> to memref<*xf32>
  call @printMemrefF32(%print_res_rsqrt) : (memref<*xf32>) -> ()

  %ret = arith.constant 0 : i32
  return %ret : i32
}
