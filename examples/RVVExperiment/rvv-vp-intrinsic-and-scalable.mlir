memref.global "private" @gv : memref<20xi32> = dense<[1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
                                                      0, 1, 1, 0, 1, 1, 1, 1, 1, 1]>

func.func private @printMemrefI32(memref<*xi32>)

func.func @alloc_mem() -> memref<20xi32> {
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

func.func @main() -> i32 {
  %mem = memref.get_global @gv : memref<20xi32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c6 = arith.constant 6 : index
  %c10 = arith.constant 10 : index
  %c1_i32 = arith.constant 1 : i32

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
  %load_vec1 = rvv.load %mem[%c0], %vl8 : memref<20xi32>, vector<[4]xi32>, index
  %load_vec2 = rvv.load %mem[%c10], %vl8 : memref<20xi32>, vector<[4]xi32>, index

  // Create the mask.
  %mask_scalable6 = vector.create_mask %vl6 : vector<[4]xi1>
  %mask_scalable8 = vector.create_mask %vl8 : vector<[4]xi1>

  //===--------------------------------------------------------------------===//
  // VP Intrinsic And Operation + Scalable Vector Type
  //===--------------------------------------------------------------------===//

  // Mask-Driven
  %res_and_mask_driven = "llvm.intr.vp.and" (%load_vec1, %load_vec2, %mask_scalable6, %vl8_i32) :
      (vector<[4]xi32>, vector<[4]xi32>, vector<[4]xi1>, i32) -> vector<[4]xi32>

  %res_and_mask_driven_mem = call @alloc_mem() : () -> memref<20xi32>
  rvv.store %res_and_mask_driven, %res_and_mask_driven_mem[%c0], %vl8 : vector<[4]xi32>, memref<20xi32>, index
  %print_res_and_mask_driven = memref.cast %res_and_mask_driven_mem : memref<20xi32> to memref<*xi32>
  call @printMemrefI32(%print_res_and_mask_driven) : (memref<*xi32>) -> ()

  // EVL-Driven
  %res_and_evl_driven = "llvm.intr.vp.and" (%load_vec1, %load_vec2, %mask_scalable8, %vl6_i32) :
      (vector<[4]xi32>, vector<[4]xi32>, vector<[4]xi1>, i32) -> vector<[4]xi32>

  %res_and_evl_driven_mem = call @alloc_mem() : () -> memref<20xi32>
  rvv.store %res_and_evl_driven, %res_and_evl_driven_mem[%c0], %vl8 : vector<[4]xi32>, memref<20xi32>, index
  %print_res_and_evl_driven = memref.cast %res_and_evl_driven_mem : memref<20xi32> to memref<*xi32>
  call @printMemrefI32(%print_res_and_evl_driven) : (memref<*xi32>) -> ()

  //===--------------------------------------------------------------------===//
  // VP Intrinsic Reduce And Operation + Scalable Vector Type
  //===--------------------------------------------------------------------===//

  // Mask-Driven
  %res_reduce_mask_driven = "llvm.intr.vp.reduce.and" (%c1_i32, %load_vec1, %mask_scalable6, %vl8_i32) :
      (i32, vector<[4]xi32>, vector<[4]xi1>, i32) -> i32
  vector.print %res_reduce_mask_driven : i32

  // EVL-Driven
  %res_reduce_evl_driven = "llvm.intr.vp.reduce.and" (%c1_i32, %load_vec1, %mask_scalable8, %vl6_i32) :
      (i32, vector<[4]xi32>, vector<[4]xi1>, i32) -> i32
  vector.print %res_reduce_evl_driven : i32

  %ret = arith.constant 0 : i32
  return %ret : i32
}
