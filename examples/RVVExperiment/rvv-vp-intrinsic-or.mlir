memref.global "private" @gv : memref<20xi32> = dense<[1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
                                                      0, 1, 1, 0, 1, 1, 1, 1, 1, 1]>

func.func @main() -> i32 {
  %mem = memref.get_global @gv : memref<20xi32>
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
  // VP Intrinsic OR Operation + Fixed Vector Type
  //===--------------------------------------------------------------------===//

  // Mask-Driven
  %vec1 = vector.load %mem[%c0] : memref<20xi32>, vector<8xi32>
  %vec2 = vector.load %mem[%c10] : memref<20xi32>, vector<8xi32>
  %res_mask_driven = "llvm.intr.vp.or" (%vec1, %vec2, %mask6, %evl8) :
         (vector<8xi32>, vector<8xi32>, vector<8xi1>, i32) -> vector<8xi32>
  vector.print %res_mask_driven : vector<8xi32>

  // EVL-Driven
  %vec3 = vector.load %mem[%c0] : memref<20xi32>, vector<8xi32>
  %vec4 = vector.load %mem[%c10] : memref<20xi32>, vector<8xi32>
  %res_evl_driven = "llvm.intr.vp.or" (%vec3, %vec4, %mask8, %evl6) :
         (vector<8xi32>, vector<8xi32>, vector<8xi1>, i32) -> vector<8xi32>
  vector.print %res_evl_driven : vector<8xi32>

  //===--------------------------------------------------------------------===//
  // VP Intrinsic Reduce OR Operation + Fixed Vector Type
  //===--------------------------------------------------------------------===//

  // Mask-Driven
  %vec5 = vector.load %mem[%c0] : memref<20xi32>, vector<8xi32>
  %res_reduce_mask_driven = "llvm.intr.vp.reduce.or" (%c1_i32, %vec5, %mask6, %evl8) :
         (i32, vector<8xi32>, vector<8xi1>, i32) -> i32
  vector.print %res_reduce_mask_driven : i32

  // EVL-Driven Error
  %vec6 = vector.load %mem[%c0] : memref<20xi32>, vector<8xi32>
  %res_reduce_evl_driven = "llvm.intr.vp.reduce.or" (%c1_i32, %vec6, %mask8, %evl6) :
         (i32, vector<8xi32>, vector<8xi1>, i32) -> i32
  vector.print %res_reduce_evl_driven : i32

  %ret = arith.constant 0 : i32
  return %ret : i32
}
