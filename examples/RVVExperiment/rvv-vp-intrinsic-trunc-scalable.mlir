memref.global "private" @gv_i64 : memref<20xi64> = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                                                          10, 11, 12, 13, 14, 15, 16, 17, 18, 19]>
memref.global "private" @gv_f32 : memref<20xf32> = dense<[0. , 1. , 2. , 3. , 4. , 5. , 6. , 7. , 8. , 9. ,
                                                          10., 11., 12., 13., 14., 15., 16., 17., 18., 19.]>
memref.global "private" @gv_f64 : memref<20xf64> = dense<[0. , 1. , 2. , 3. , 4. , 5. , 6. , 7. , 8. , 9. ,
                                                          10., 11., 12., 13., 14., 15., 16., 17., 18., 19.]>

func.func @main() -> i32 {
  %mem_i64 = memref.get_global @gv_i64 : memref<20xi64>
  %mem_f32 = memref.get_global @gv_f32 : memref<20xf32>
  %mem_f64 = memref.get_global @gv_f64 : memref<20xf64>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c6 = arith.constant 6 : index
  %c10 = arith.constant 10 : index

  // Configure the register.
  // SEW = 32
  %sew = arith.constant 2 : index
  // LMUL = 2
  %lmul = arith.constant 1 : index
  // AVL = 6 / 8
  %avl6 = arith.constant 6 : index
  %avl8 = arith.constant 8 : index

  // Load vl elements.
  %vl6 = rvv.setvl %avl6, %sew, %lmul : index
  %vl6_i32 = arith.index_cast %vl6 : index to i32
  %vl8 = rvv.setvl %avl8, %sew, %lmul : index
  %vl8_i32 = arith.index_cast %vl8 : index to i32
  %load_vec1_i64 = rvv.load %mem_i64[%c0], %vl8 : memref<20xi64>, vector<[4]xi64>, index
  %load_vec2_i64 = rvv.load %mem_i64[%c10], %vl8 : memref<20xi64>, vector<[4]xi64>, index
  %load_vec1_f64 = rvv.load %mem_f64[%c0], %vl8 : memref<20xf64>, vector<[4]xf64>, index
  %load_vec2_f64 = rvv.load %mem_f64[%c10], %vl8 : memref<20xf64>, vector<[4]xf64>, index

  // Create the mask.
  %mask_scalable6 = vector.create_mask %vl6 : vector<[4]xi1>
  %mask_scalable8 = vector.create_mask %vl8 : vector<[4]xi1>

  //===--------------------------------------------------------------------===//
  // VP Intrinsic Trunc Operation + Scalable Vector Type
  //===--------------------------------------------------------------------===//

  // Mask-Driven
  %res_trunc_mask_driven = "llvm.intr.vp.trunc" (%load_vec1_i64, %mask_scalable6, %vl8_i32) :
      (vector<[4]xi64>, vector<[4]xi1>, i32) -> vector<[4]xi32>
  call @print_scalable_vector_i32(%res_trunc_mask_driven) : (vector<[4]xi32>) -> ()

  // EVL-Driven
  %res_trunc_evl_driven = "llvm.intr.vp.trunc" (%load_vec1_i64, %mask_scalable8, %vl6_i32) :
      (vector<[4]xi64>, vector<[4]xi1>, i32) -> vector<[4]xi32>
  call @print_scalable_vector_i32(%res_trunc_evl_driven) : (vector<[4]xi32>) -> ()

  //===--------------------------------------------------------------------===//
  // VP Intrinsic FPTrunc Operation + Scalable Vector Type
  //===--------------------------------------------------------------------===//

  // Mask-Driven
  %res_fptrunc_mask_driven = "llvm.intr.vp.fptrunc" (%load_vec1_f64, %mask_scalable6, %vl8_i32) :
      (vector<[4]xf64>, vector<[4]xi1>, i32) -> vector<[4]xf32>
  call @print_scalable_vector_f32(%res_fptrunc_mask_driven) : (vector<[4]xf32>) -> ()

  // EVL-Driven
  %res_fptrunc_evl_driven = "llvm.intr.vp.fptrunc" (%load_vec1_f64, %mask_scalable8, %vl6_i32) :
      (vector<[4]xf64>, vector<[4]xi1>, i32) -> vector<[4]xf32>
  call @print_scalable_vector_f32(%res_fptrunc_evl_driven) : (vector<[4]xf32>) -> ()

  %ret = arith.constant 0 : i32
  return %ret : i32
}

func.func private @printMemrefF32(memref<*xf32>)
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

func.func @print_scalable_vector_f32(%vec : vector<[4]xf32>) {
  %c0 = arith.constant 0 : index
  %vl8 = arith.constant 8 : index
  %res_mem = call @alloc_mem_f32() : () -> memref<20xf32>
  rvv.store %vec, %res_mem[%c0], %vl8 : vector<[4]xf32>, memref<20xf32>, index
  %print_vec = memref.cast %res_mem : memref<20xf32> to memref<*xf32>
  call @printMemrefF32(%print_vec) : (memref<*xf32>) -> ()
  return
}

func.func @print_scalable_vector_i32(%vec : vector<[4]xi32>) {
  %c0 = arith.constant 0 : index
  %vl8 = arith.constant 8 : index
  %res_mem = call @alloc_mem_i32() : () -> memref<20xi32>
  rvv.store %vec, %res_mem[%c0], %vl8 : vector<[4]xi32>, memref<20xi32>, index
  %print_vec = memref.cast %res_mem : memref<20xi32> to memref<*xi32>
  call @printMemrefI32(%print_vec) : (memref<*xi32>) -> ()
  return
}
