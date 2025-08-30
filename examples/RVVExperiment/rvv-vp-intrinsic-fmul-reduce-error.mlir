memref.global "private" @gv_f32 : memref<20xf32> = dense<[0. , 1. , 2. , 3. , 4. , 5. , 6. , 7. , 8. , 9. ,
                                                          10., 11., 12., 13., 14., 15., 16., 17., 18., 19.]>
func.func @main() -> i32 {
  %mem_f32 = memref.get_global @gv_f32 : memref<20xf32>
  %c0 = arith.constant 0 : index
  %mask6 = arith.constant dense<[1, 1, 1, 1, 1, 1, 0, 0]> : vector<8xi1>
  %evl8 = arith.constant 8 : i32
  %mask8 = arith.constant dense<[1, 1, 1, 1, 1, 1, 1, 1]> : vector<8xi1>
  %evl6 = arith.constant 6 : i32
  %c1_f32 = arith.constant 1.0 : f32

  //===--------------------------------------------------------------------===//
  // VP Intrinsic Reduce FMul Operation + Fixed Vector Type
  //===--------------------------------------------------------------------===//

  // Mask-Driven Error
  %vec1 = vector.load %mem_f32[%c0] : memref<20xf32>, vector<8xf32>
  %res_reduce_fmul_mask_driven = "llvm.intr.vp.reduce.fmul" (%c1_f32, %vec1, %mask6, %evl8) :
         (f32, vector<8xf32>, vector<8xi1>, i32) -> f32
  vector.print %res_reduce_fmul_mask_driven : f32

  // EVL-Driven Error
  %vec2 = vector.load %mem_f32[%c0] : memref<20xf32>, vector<8xf32>
  %res_reduce_fmul_evl_driven = "llvm.intr.vp.reduce.fmul" (%c1_f32, %vec2, %mask8, %evl6) :
         (f32, vector<8xf32>, vector<8xi1>, i32) -> f32
  vector.print %res_reduce_fmul_evl_driven : f32

  %ret = arith.constant 0 : i32
  return %ret : i32
}
