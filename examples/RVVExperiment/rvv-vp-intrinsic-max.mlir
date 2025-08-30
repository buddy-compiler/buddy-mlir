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
  %c1_f32 = arith.constant 1.0 : f32

  //===--------------------------------------------------------------------===//
  // VP Intrinsic FMax Reduce Operation + Fixed Vector Type
  //===--------------------------------------------------------------------===//

  // Mask-Driven
  %vec1 = vector.load %mem_f32[%c0] : memref<20xf32>, vector<8xf32>
  %res_fmax_mask_driven = "llvm.intr.vp.reduce.fmax" (%c1_f32, %vec1, %mask6, %evl8) :
         (f32, vector<8xf32>, vector<8xi1>, i32) -> f32
  vector.print %res_fmax_mask_driven : f32

  // EVL-Driven
  %vec2 = vector.load %mem_f32[%c0] : memref<20xf32>, vector<8xf32>
  %res_fmax_evl_driven = "llvm.intr.vp.reduce.fmax" (%c1_f32, %vec2, %mask8, %evl6) :
         (f32, vector<8xf32>, vector<8xi1>, i32) -> f32
  vector.print %res_fmax_evl_driven : f32

  //===--------------------------------------------------------------------===//
  // VP Intrinsic SMax Reduce Operation + Fixed Vector Type
  //===--------------------------------------------------------------------===//

  // Mask-Driven
  %vec3 = vector.load %mem_i32[%c0] : memref<20xi32>, vector<8xi32>
  %res_smax_mask_driven = "llvm.intr.vp.reduce.smax" (%c1_i32, %vec3, %mask6, %evl8) :
         (i32, vector<8xi32>, vector<8xi1>, i32) -> i32
  vector.print %res_smax_mask_driven : i32

  // EVL-Driven
  %vec4 = vector.load %mem_i32[%c0] : memref<20xi32>, vector<8xi32>
  %res_smax_evl_driven = "llvm.intr.vp.reduce.smax" (%c1_i32, %vec4, %mask8, %evl6) :
         (i32, vector<8xi32>, vector<8xi1>, i32) -> i32
  vector.print %res_smax_evl_driven : i32

  //===--------------------------------------------------------------------===//
  // VP Intrinsic UMax Reduce Operation + Fixed Vector Type
  //===--------------------------------------------------------------------===//

  // Mask-Driven
  %vec5 = vector.load %mem_i32[%c0] : memref<20xi32>, vector<8xi32>
  %res_umax_mask_driven = "llvm.intr.vp.reduce.umax" (%c1_i32, %vec5, %mask6, %evl8) :
         (i32, vector<8xi32>, vector<8xi1>, i32) -> i32
  vector.print %res_umax_mask_driven : i32

  // EVL-Driven
  %vec6 = vector.load %mem_i32[%c0] : memref<20xi32>, vector<8xi32>
  %res_umax_evl_driven = "llvm.intr.vp.reduce.umax" (%c1_i32, %vec6, %mask8, %evl6) :
         (i32, vector<8xi32>, vector<8xi1>, i32) -> i32
  vector.print %res_umax_evl_driven : i32

  %ret = arith.constant 0 : i32
  return %ret : i32
}
