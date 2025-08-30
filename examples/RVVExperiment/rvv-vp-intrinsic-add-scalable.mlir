memref.global "private" @gv_i32 : memref<20xi32> = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                                                          10, 11, 12, 13, 14, 15, 16, 17, 18, 19]>
memref.global "private" @gv_f32 : memref<20xf32> = dense<[0. , 1. , 2. , 3. , 4. , 5. , 6. , 7. , 8. , 9. ,
                                                          10., 11., 12., 13., 14., 15., 16., 17., 18., 19.]>

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

func.func @main() -> i32 {
  %mem_i32 = memref.get_global @gv_i32 : memref<20xi32>
  %mem_f32 = memref.get_global @gv_f32 : memref<20xf32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c6 = arith.constant 6 : index
  %c10 = arith.constant 10 : index
  %mask6 = arith.constant dense<[1, 1, 1, 1, 1, 1, 0, 0]> : vector<8xi1>
  %evl8 = arith.constant 8 : i32
  %mask8 = arith.constant dense<[1, 1, 1, 1, 1, 1, 1, 1]> : vector<8xi1>
  %evl6 = arith.constant 6 : i32
  %c1_i32 = arith.constant 1 : i32
  %c1_f32 = arith.constant 1.0 : f32

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
  %load_vec1_i32 = rvv.load %mem_i32[%c0], %vl8 : memref<20xi32>, vector<[4]xi32>, index
  %load_vec2_i32 = rvv.load %mem_i32[%c10], %vl8 : memref<20xi32>, vector<[4]xi32>, index
  %load_vec1_f32 = rvv.load %mem_f32[%c0], %vl8 : memref<20xf32>, vector<[4]xf32>, index
  %load_vec2_f32 = rvv.load %mem_f32[%c10], %vl8 : memref<20xf32>, vector<[4]xf32>, index

  // Create the mask.
  %mask_scalable6 = vector.create_mask %vl6 : vector<[4]xi1>
  %mask_scalable8 = vector.create_mask %vl8 : vector<[4]xi1>

  //===--------------------------------------------------------------------===//
  // VP Intrinsic Add Operation + Scalable Vector Type
  //===--------------------------------------------------------------------===//

  // Mask-Driven
  %res_add_mask_driven = "llvm.intr.vp.add" (%load_vec2_i32, %load_vec1_i32, %mask_scalable6, %vl8_i32) :
      (vector<[4]xi32>, vector<[4]xi32>, vector<[4]xi1>, i32) -> vector<[4]xi32>

  %res_add_mask_driven_mem = call @alloc_mem_i32() : () -> memref<20xi32>
  rvv.store %res_add_mask_driven, %res_add_mask_driven_mem[%c0], %vl8 : vector<[4]xi32>, memref<20xi32>, index
  %print_res_add_mask_driven = memref.cast %res_add_mask_driven_mem : memref<20xi32> to memref<*xi32>
  call @printMemrefI32(%print_res_add_mask_driven) : (memref<*xi32>) -> ()

  // EVL-Driven
  %res_add_evl_driven = "llvm.intr.vp.add" (%load_vec2_i32, %load_vec1_i32, %mask_scalable8, %vl6_i32) :
      (vector<[4]xi32>, vector<[4]xi32>, vector<[4]xi1>, i32) -> vector<[4]xi32>

  %res_add_evl_driven_mem = call @alloc_mem_i32() : () -> memref<20xi32>
  rvv.store %res_add_evl_driven, %res_add_evl_driven_mem[%c0], %vl8 : vector<[4]xi32>, memref<20xi32>, index
  %print_res_add_evl_driven = memref.cast %res_add_evl_driven_mem : memref<20xi32> to memref<*xi32>
  call @printMemrefI32(%print_res_add_evl_driven) : (memref<*xi32>) -> ()

  //===--------------------------------------------------------------------===//
  // VP Intrinsic FAdd Operation + Scalable Vector Type
  //===--------------------------------------------------------------------===//

  // Mask-Driven
  %res_fadd_mask_driven = "llvm.intr.vp.fadd" (%load_vec2_f32, %load_vec1_f32, %mask_scalable6, %vl8_i32) :
      (vector<[4]xf32>, vector<[4]xf32>, vector<[4]xi1>, i32) -> vector<[4]xf32>

  %res_fadd_mask_driven_mem = call @alloc_mem_f32() : () -> memref<20xf32>
  rvv.store %res_fadd_mask_driven, %res_fadd_mask_driven_mem[%c0], %vl8 : vector<[4]xf32>, memref<20xf32>, index
  %print_res_fadd_mask_driven = memref.cast %res_fadd_mask_driven_mem : memref<20xf32> to memref<*xf32>
  call @printMemrefF32(%print_res_fadd_mask_driven) : (memref<*xf32>) -> ()

  // EVL-Driven
  %res_fadd_evl_driven = "llvm.intr.vp.fadd" (%load_vec2_f32, %load_vec1_f32, %mask_scalable8, %vl6_i32) :
      (vector<[4]xf32>, vector<[4]xf32>, vector<[4]xi1>, i32) -> vector<[4]xf32>

  %res_fadd_evl_driven_mem = call @alloc_mem_f32() : () -> memref<20xf32>
  rvv.store %res_fadd_evl_driven, %res_fadd_evl_driven_mem[%c0], %vl8 : vector<[4]xf32>, memref<20xf32>, index
  %print_res_fadd_evl_driven = memref.cast %res_fadd_evl_driven_mem : memref<20xf32> to memref<*xf32>
  call @printMemrefF32(%print_res_fadd_evl_driven) : (memref<*xf32>) -> ()

  //===--------------------------------------------------------------------===//
  // VP Intrinsic Reduce Add Operation + Scalable Vector Type
  //===--------------------------------------------------------------------===//

  // Mask-Driven
  %res_reduce_add_mask_driven = "llvm.intr.vp.reduce.add" (%c1_i32, %load_vec1_i32, %mask_scalable6, %vl8_i32) :
      (i32, vector<[4]xi32>, vector<[4]xi1>, i32) -> i32
  vector.print %res_reduce_add_mask_driven : i32

  // EVL-Driven
  %res_reduce_add_evl_driven = "llvm.intr.vp.reduce.add" (%c1_i32, %load_vec1_i32, %mask_scalable8, %vl6_i32) :
      (i32, vector<[4]xi32>, vector<[4]xi1>, i32) -> i32
  vector.print %res_reduce_add_evl_driven : i32

  //===--------------------------------------------------------------------===//
  // VP Intrinsic Reduce FAdd Operation + Scalable Vector Type
  //===--------------------------------------------------------------------===//

  // Mask-Driven
  %res_reduce_fadd_mask_driven = "llvm.intr.vp.reduce.fadd" (%c1_f32, %load_vec1_f32, %mask_scalable6, %vl8_i32) :
      (f32, vector<[4]xf32>, vector<[4]xi1>, i32) -> f32
  vector.print %res_reduce_fadd_mask_driven : f32

  // EVL-Driven
  %res_reduce_fadd_evl_driven = "llvm.intr.vp.reduce.fadd" (%c1_f32, %load_vec1_f32, %mask_scalable8, %vl6_i32) :
      (f32, vector<[4]xf32>, vector<[4]xi1>, i32) -> f32
  vector.print %res_reduce_fadd_evl_driven : f32

  %ret = arith.constant 0 : i32
  return %ret : i32
}
