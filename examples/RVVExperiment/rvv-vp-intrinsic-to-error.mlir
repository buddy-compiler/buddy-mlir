memref.global "private" @gv_i32 : memref<20xi32> = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                                                          10, 11, 12, 13, 14, 15, 16, 17, 18, 19]>
func.func @main() -> i32 {
  %mem_i32 = memref.get_global @gv_i32 : memref<20xi32>
  %c0 = arith.constant 0 : index
  %mask6 = arith.constant dense<[1, 1, 1, 1, 1, 1, 0, 0]> : vector<8xi1>
  %evl8 = arith.constant 8 : i32
  %mask8 = arith.constant dense<[1, 1, 1, 1, 1, 1, 1, 1]> : vector<8xi1>
  %evl6 = arith.constant 6 : i32

  //===--------------------------------------------------------------------===//
  // VP Intrinsic IntToPtr Operation + Fixed Vector Type
  // VP Intrinsic PtrToInt Operation + Fixed Vector Type
  //===--------------------------------------------------------------------===//

  // Mask-Driven Error
  %vec1 = vector.load %mem_i32[%c0] : memref<20xi32>, vector<8xi32>
  %res_inttoptr_mask_driven = "llvm.intr.vp.inttoptr" (%vec1, %mask6, %evl8) :
         (vector<8xi32>, vector<8xi1>, i32) -> !llvm.vec<8 x !llvm.ptr<i32>>
  %res_ptrtoint_mask_driven = "llvm.intr.vp.ptrtoint" (%res_inttoptr_mask_driven, %mask6, %evl8) :
         (!llvm.vec<8 x !llvm.ptr<i32>>, vector<8xi1>, i32) -> vector<8xi32>
  vector.print %res_ptrtoint_mask_driven : vector<8xi32>

  // EVL-Driven Error
  %vec2 = vector.load %mem_i32[%c0] : memref<20xi32>, vector<8xi32>
  %res_inttoptr_evl_driven = "llvm.intr.vp.inttoptr" (%vec2, %mask8, %evl6) :
         (vector<8xi32>, vector<8xi1>, i32) -> !llvm.vec<8 x !llvm.ptr<i32>>
  %res_ptrtoint_evl_driven = "llvm.intr.vp.ptrtoint" (%res_inttoptr_evl_driven, %mask8, %evl6) :
         (!llvm.vec<8 x !llvm.ptr<i32>>, vector<8xi1>, i32) -> vector<8xi32>
  vector.print %res_ptrtoint_evl_driven : vector<8xi32>

  %ret = arith.constant 0 : i32
  return %ret : i32
}
