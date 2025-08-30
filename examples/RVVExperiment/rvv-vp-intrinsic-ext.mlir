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
  %mask6 = arith.constant dense<[1, 1, 1, 1, 1, 1, 0, 0]> : vector<8xi1>
  %evl8 = arith.constant 8 : i32
  %mask8 = arith.constant dense<[1, 1, 1, 1, 1, 1, 1, 1]> : vector<8xi1>
  %evl6 = arith.constant 6 : i32
  %c1_i32 = arith.constant 1 : i32

  //===--------------------------------------------------------------------===//
  // VP Intrinsic FPExt Operation + Fixed Vector Type
  //===--------------------------------------------------------------------===//

  // Mask-Driven
  %vec1 = vector.load %mem_f32[%c0] : memref<20xf32>, vector<8xf32>
  %res_fpext_mask_driven = "llvm.intr.vp.fpext" (%vec1, %mask6, %evl8) :
         (vector<8xf32>, vector<8xi1>, i32) -> vector<8xf64>
  vector.print %res_fpext_mask_driven : vector<8xf64>

  // EVL-Driven
  %vec2 = vector.load %mem_f32[%c0] : memref<20xf32>, vector<8xf32>
  %res_fpext_evl_driven = "llvm.intr.vp.fpext" (%vec2, %mask8, %evl6) :
         (vector<8xf32>, vector<8xi1>, i32) -> vector<8xf64>
  vector.print %res_fpext_evl_driven : vector<8xf64>

  //===--------------------------------------------------------------------===//
  // VP Intrinsic SExt Operation + Fixed Vector Type
  //===--------------------------------------------------------------------===//

  // Mask-Driven
  %vec3 = vector.load %mem_i32[%c0] : memref<20xi32>, vector<8xi32>
  %res_sext_mask_driven = "llvm.intr.vp.sext" (%vec3, %mask6, %evl8) :
         (vector<8xi32>, vector<8xi1>, i32) -> vector<8xi64>
  vector.print %res_sext_mask_driven : vector<8xi64>

  // EVL-Driven
  %vec4 = vector.load %mem_i32[%c0] : memref<20xi32>, vector<8xi32>
  %res_sext_evl_driven = "llvm.intr.vp.sext" (%vec4, %mask8, %evl6) :
         (vector<8xi32>, vector<8xi1>, i32) -> vector<8xi64>
  vector.print %res_sext_evl_driven : vector<8xi64>

  //===--------------------------------------------------------------------===//
  // VP Intrinsic ZExt Operation + Fixed Vector Type
  //===--------------------------------------------------------------------===//

  // Mask-Driven
  %vec5 = vector.load %mem_i32[%c0] : memref<20xi32>, vector<8xi32>
  %res_zext_mask_driven = "llvm.intr.vp.zext" (%vec5, %mask6, %evl8) :
         (vector<8xi32>, vector<8xi1>, i32) -> vector<8xi64>
  vector.print %res_zext_mask_driven : vector<8xi64>

  // EVL-Driven
  %vec6 = vector.load %mem_i32[%c0] : memref<20xi32>, vector<8xi32>
  %res_zext_evl_driven = "llvm.intr.vp.zext" (%vec6, %mask8, %evl6) :
         (vector<8xi32>, vector<8xi1>, i32) -> vector<8xi64>
  vector.print %res_zext_evl_driven : vector<8xi64>

  %ret = arith.constant 0 : i32
  return %ret : i32
}
