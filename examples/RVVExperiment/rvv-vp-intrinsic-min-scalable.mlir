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
  %load_vec_i32 = rvv.load %mem_i32[%c0], %vl8 : memref<20xi32>, vector<[4]xi32>, index
  %load_vec_f32 = rvv.load %mem_f32[%c0], %vl8 : memref<20xf32>, vector<[4]xf32>, index

  // Create the mask.
  %mask_scalable6 = vector.create_mask %vl6 : vector<[4]xi1>
  %mask_scalable8 = vector.create_mask %vl8 : vector<[4]xi1>

  //===--------------------------------------------------------------------===//
  // VP Intrinsic FMin Reduce Operation + Scalable Vector Type
  //===--------------------------------------------------------------------===//

  // Mask-Driven
  %res_fmin_mask_driven = "llvm.intr.vp.reduce.fmin" (%c1_f32, %load_vec_f32, %mask_scalable6, %vl8_i32) :
      (f32, vector<[4]xf32>, vector<[4]xi1>, i32) -> f32
  vector.print %res_fmin_mask_driven : f32

  // EVL-Driven
  %res_fmin_evl_driven = "llvm.intr.vp.reduce.fmin" (%c1_f32, %load_vec_f32, %mask_scalable8, %vl6_i32) :
      (f32, vector<[4]xf32>, vector<[4]xi1>, i32) -> f32
  vector.print %res_fmin_evl_driven : f32

  //===--------------------------------------------------------------------===//
  // VP Intrinsic SMin Reduce Operation + Scalable Vector Type
  //===--------------------------------------------------------------------===//

  // Mask-Driven
  %res_smin_mask_driven = "llvm.intr.vp.reduce.smin" (%c1_i32, %load_vec_i32, %mask_scalable6, %vl8_i32) :
      (i32, vector<[4]xi32>, vector<[4]xi1>, i32) -> i32
  vector.print %res_smin_mask_driven : i32

  // EVL-Driven
  %res_smin_evl_driven = "llvm.intr.vp.reduce.smin" (%c1_i32, %load_vec_i32, %mask_scalable8, %vl6_i32) :
      (i32, vector<[4]xi32>, vector<[4]xi1>, i32) -> i32
  vector.print %res_smin_evl_driven : i32

  //===--------------------------------------------------------------------===//
  // VP Intrinsic UMax Reduce Operation + Scalable Vector Type
  //===--------------------------------------------------------------------===//

  // Mask-Driven
  %res_umin_mask_driven = "llvm.intr.vp.reduce.umin" (%c1_i32, %load_vec_i32, %mask_scalable6, %vl8_i32) :
      (i32, vector<[4]xi32>, vector<[4]xi1>, i32) -> i32
  vector.print %res_umin_mask_driven : i32

  // EVL-Driven
  %vec6 = vector.load %mem_i32[%c0] : memref<20xi32>, vector<8xi32>
  %res_umin_evl_driven = "llvm.intr.vp.reduce.umin" (%c1_i32, %load_vec_i32, %mask_scalable8, %vl6_i32) :
      (i32, vector<[4]xi32>, vector<[4]xi1>, i32) -> i32
  vector.print %res_umin_evl_driven : i32

  %ret = arith.constant 0 : i32
  return %ret : i32
}
