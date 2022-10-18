memref.global "private" @gv_i32 : memref<20xi32> = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                                                          10, 11, 12, 13, 14, 15, 16, 17, 18, 19]>
memref.global "private" @gv_f32 : memref<20xf32> = dense<[0. , 1. , 2. , 3. , 4. , 5. , 6. , 7. , 8. , 9. ,
                                                          10., 11., 12., 13., 14., 15., 16., 17., 18., 19.]>

func.func @main() -> i32 {
  %mem_i32 = memref.get_global @gv_i32 : memref<20xi32>
  %mem_f32 = memref.get_global @gv_f32 : memref<20xf32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c6 = arith.constant 6 : index
  %c10 = arith.constant 10 : index
  %c1_i32 = arith.constant 1 : i32

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
  %load_vec_i32 = rvv.load %mem_i32[%c0], %vl8 : memref<20xi32>, vector<[4]xi32>, index
  %load_vec_f32 = rvv.load %mem_f32[%c0], %vl8 : memref<20xf32>, vector<[4]xf32>, index

  // Create the mask.
  %mask_scalable6 = vector.create_mask %vl6 : vector<[4]xi1>
  %mask_scalable8 = vector.create_mask %vl8 : vector<[4]xi1>

  //===--------------------------------------------------------------------===//
  // VP Intrinsic FPExt Operation + Fixed Vector Type
  //===--------------------------------------------------------------------===//

  // Mask-Driven
  %res_fpext_mask_driven = "llvm.intr.vp.fpext" (%load_vec_f32, %mask_scalable6, %vl8_i32) :
      (vector<[4]xf32>, vector<[4]xi1>, i32) -> vector<[4]xf64>
  call @print_scalable_vector_f64(%res_fpext_mask_driven) : (vector<[4]xf64>) -> ()

  // EVL-Driven
  %res_fpext_evl_driven = "llvm.intr.vp.fpext" (%load_vec_f32, %mask_scalable8, %vl6_i32) :
      (vector<[4]xf32>, vector<[4]xi1>, i32) -> vector<[4]xf64>
  call @print_scalable_vector_f64(%res_fpext_evl_driven) : (vector<[4]xf64>) -> ()

  //===--------------------------------------------------------------------===//
  // VP Intrinsic SExt Operation + Fixed Vector Type
  //===--------------------------------------------------------------------===//

  // Mask-Driven
  %res_sext_mask_driven = "llvm.intr.vp.sext" (%load_vec_i32, %mask_scalable6, %vl8_i32) :
      (vector<[4]xi32>, vector<[4]xi1>, i32) -> vector<[4]xi64>
  call @print_scalable_vector_i64(%res_sext_mask_driven) : (vector<[4]xi64>) -> ()

  // EVL-Driven
  %res_sext_evl_driven = "llvm.intr.vp.sext" (%load_vec_i32, %mask_scalable8, %vl6_i32) :
      (vector<[4]xi32>, vector<[4]xi1>, i32) -> vector<[4]xi64>
  call @print_scalable_vector_i64(%res_sext_evl_driven) : (vector<[4]xi64>) -> ()

  //===--------------------------------------------------------------------===//
  // VP Intrinsic ZExt Operation + Fixed Vector Type
  //===--------------------------------------------------------------------===//

  // Mask-Driven
  %res_zext_mask_driven = "llvm.intr.vp.zext" (%load_vec_i32, %mask_scalable6, %vl8_i32) :
      (vector<[4]xi32>, vector<[4]xi1>, i32) -> vector<[4]xi64>
  call @print_scalable_vector_i64(%res_zext_mask_driven) : (vector<[4]xi64>) -> ()

  // EVL-Driven
  %res_zext_evl_driven = "llvm.intr.vp.zext" (%load_vec_i32, %mask_scalable8, %vl6_i32) :
      (vector<[4]xi32>, vector<[4]xi1>, i32) -> vector<[4]xi64>
  call @print_scalable_vector_i64(%res_zext_evl_driven) : (vector<[4]xi64>) -> ()

  %ret = arith.constant 0 : i32
  return %ret : i32
}

func.func private @printMemrefF64(memref<*xf64>)
func.func private @printMemrefI64(memref<*xi64>)

func.func @alloc_mem_i64() -> memref<20xi64> {
  %i0 = arith.constant 0 : i64
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %mem = memref.alloc() : memref<20xi64>
  %dim = memref.dim %mem, %c0 : memref<20xi64>
  scf.for %idx = %c0 to %dim step %c1 {
    memref.store %i0, %mem[%idx] : memref<20xi64>
  }
  return %mem : memref<20xi64>
}

func.func @alloc_mem_f64() -> memref<20xf64> {
  %f0 = arith.constant 0.0 : f64
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %mem = memref.alloc() : memref<20xf64>
  %dim = memref.dim %mem, %c0 : memref<20xf64>
  scf.for %idx = %c0 to %dim step %c1 {
    memref.store %f0, %mem[%idx] : memref<20xf64>
  }
  return %mem : memref<20xf64>
}

func.func @print_scalable_vector_f64(%vec : vector<[4]xf64>) {
  %c0 = arith.constant 0 : index
  %vl8 = arith.constant 8 : index
  %res_mem = call @alloc_mem_f64() : () -> memref<20xf64>
  rvv.store %vec, %res_mem[%c0], %vl8 : vector<[4]xf64>, memref<20xf64>, index
  %print_vec = memref.cast %res_mem : memref<20xf64> to memref<*xf64>
  call @printMemrefF64(%print_vec) : (memref<*xf64>) -> ()
  return
}

func.func @print_scalable_vector_i64(%vec : vector<[4]xi64>) {
  %c0 = arith.constant 0 : index
  %vl8 = arith.constant 8 : index
  %res_mem = call @alloc_mem_i64() : () -> memref<20xi64>
  rvv.store %vec, %res_mem[%c0], %vl8 : vector<[4]xi64>, memref<20xi64>, index
  %print_vec = memref.cast %res_mem : memref<20xi64> to memref<*xi64>
  call @printMemrefI64(%print_vec) : (memref<*xi64>) -> ()
  return
}
