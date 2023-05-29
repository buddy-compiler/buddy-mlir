memref.global "private" @gv_i32 : memref<20xi32> = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                                                          10, 11, 12, 13, 14, 15, 16, 17, 18, 19]>
func.func @main() -> i32 {
  %mem_i32 = memref.get_global @gv_i32 : memref<20xi32>
  %c0 = arith.constant 0 : index
  %mask6 = arith.constant dense<[1, 1, 1, 1, 1, 1, 0, 0]> : vector<8xi1>
  %evl8 = arith.constant 8 : i32
  %mask8 = arith.constant dense<[1, 1, 1, 1, 1, 1, 1, 1]> : vector<8xi1>
  %evl6 = arith.constant 6 : i32
  %c1_i32 = arith.constant 1 : i32

  //===--------------------------------------------------------------------===//
  // VP Intrinsic Reduce Mul Operation + Fixed Vector Type
  //===--------------------------------------------------------------------===//

  // Mask-Driven
  %vec1 = vector.load %mem_i32[%c0] : memref<20xi32>, vector<8xi32>
  %res_reduce_mul_mask_driven = "llvm.intr.vp.reduce.mul" (%c1_i32, %vec1, %mask6, %evl8) :
         (i32, vector<8xi32>, vector<8xi1>, i32) -> i32
  vector.print %res_reduce_mul_mask_driven : i32

  // EVL-Driven
  %vec2 = vector.load %mem_i32[%c0] : memref<20xi32>, vector<8xi32>
  %res_reduce_mul_evl_driven = "llvm.intr.vp.reduce.mul" (%c1_i32, %vec2, %mask8, %evl6) :
         (i32, vector<8xi32>, vector<8xi1>, i32) -> i32
  vector.print %res_reduce_mul_evl_driven : i32

  %ret = arith.constant 0 : i32
  return %ret : i32
}
