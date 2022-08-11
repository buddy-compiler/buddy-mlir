memref.global "private" @gv_f32 : memref<20xf32> = dense<[0. , 1. , 2. , 3. , 4. , 5. , 6. , 7. , 8. , 9. ,
                                                          10., 11., 12., 13., 14., 15., 16., 17., 18., 19.]>
func.func @main() -> i32 {
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
  // VP Intrinsic FMA F32 Operation + Fixed Vector Type
  //===--------------------------------------------------------------------===//

  // Mask-Driven
  %vec1 = vector.load %mem_f32[%c0] : memref<20xf32>, vector<8xf32>
  %vec2 = vector.load %mem_f32[%c10] : memref<20xf32>, vector<8xf32>
  %res_fma_mask_driven = "llvm.intr.vp.fma" (%vec2, %vec1, %vec1, %mask6, %evl8) :
         (vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xi1>, i32) -> vector<8xf32>
  vector.print %res_fma_mask_driven : vector<8xf32>

  // EVL-Driven
  %vec3 = vector.load %mem_f32[%c0] : memref<20xf32>, vector<8xf32>
  %vec4 = vector.load %mem_f32[%c10] : memref<20xf32>, vector<8xf32>
  %res_fma_evl_driven = "llvm.intr.vp.fma" (%vec4, %vec3, %vec3, %mask8, %evl6) :
         (vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xi1>, i32) -> vector<8xf32>
  vector.print %res_fma_evl_driven : vector<8xf32>

  %ret = arith.constant 0 : i32
  return %ret : i32
}
