memref.global "private" @gv_f32 : memref<20xf32> = dense<[0. , 1. , 2. , 3. , 4. , 5. , 6. , 7. , 8. , 9. ,
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
  %mem_f32 = memref.get_global @gv_f32 : memref<20xf32>
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
  %load_vec = rvv.load %mem_f32[%c0], %vl8 : memref<20xf32>, vector<[4]xf32>, index

  // Create the mask.
  %mask_scalable6 = vector.create_mask %vl6 : vector<[4]xi1>
  %mask_scalable8 = vector.create_mask %vl8 : vector<[4]xi1>

  //===--------------------------------------------------------------------===//
  // VP Intrinsic FNeg Operation + Scalable Vector Type
  //===--------------------------------------------------------------------===//

  // Mask-Driven
  %res_fneg_mask_driven = "llvm.intr.vp.fneg" (%load_vec, %mask_scalable6, %vl8_i32) :
       (vector<[4]xf32>, vector<[4]xi1>, i32) -> vector<[4]xf32>

  %res_mask_driven = call @alloc_mem() : () -> memref<20xf32>
  rvv.store %res_fneg_mask_driven, %res_mask_driven[%c0], %vl8 : vector<[4]xf32>, memref<20xf32>, index
  %print_res_mask_driven = memref.cast %res_mask_driven : memref<20xf32> to memref<*xf32>
  call @printMemrefF32(%print_res_mask_driven) : (memref<*xf32>) -> ()

  // EVL-Driven
  %res_fneg_evl_driven = "llvm.intr.vp.fneg" (%load_vec, %mask_scalable8, %vl6_i32) :
       (vector<[4]xf32>, vector<[4]xi1>, i32) -> vector<[4]xf32>
  
  %res_evl_driven = call @alloc_mem() : () -> memref<20xf32>
  rvv.store %res_fneg_evl_driven, %res_evl_driven[%c0], %vl8 : vector<[4]xf32>, memref<20xf32>, index
  %print_res_evl_driven = memref.cast %res_evl_driven : memref<20xf32> to memref<*xf32>
  call @printMemrefF32(%print_res_evl_driven) : (memref<*xf32>) -> ()

  %ret = arith.constant 0 : i32
  return %ret : i32
}
