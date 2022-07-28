memref.global "private" @gv : memref<20xf32> = dense<[0. , 1. , 2. , 3. , 4. , 5. , 6. , 7. , 8. , 9. ,
                                                      10., 11., 12., 13., 14., 15., 16., 17., 18., 19.]>

func.func private @printMemrefF32(memref<*xf32>)

func.func @alloc_mem() -> memref<8xf32> {
  %f0 = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %mem = memref.alloc() : memref<8xf32>
  %dim = memref.dim %mem, %c0 : memref<8xf32>
  scf.for %idx = %c0 to %dim step %c1 {
    memref.store %f0, %mem[%idx] : memref<8xf32>
  }
  return %mem : memref<8xf32>
}

func.func @main() -> i32 {
  %mem = memref.get_global @gv : memref<20xf32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c6 = arith.constant 6 : index
  %c10 = arith.constant 10 : index
  %mask = arith.constant dense<[1, 1, 1, 1, 1, 1, 0, 0]> : vector<8xi1>
  %evl = arith.constant 8 : i32

  //===--------------------------------------------------------------------===//
  // VP Intrinsic Add Operation + Fixed Vector Type
  //===--------------------------------------------------------------------===//
  
  %vec1 = vector.load %mem[%c0] : memref<20xf32>, vector<8xf32>
  %vec2 = vector.load %mem[%c10] : memref<20xf32>, vector<8xf32>
  %res = "llvm.intr.vp.fadd" (%vec1, %vec2, %mask, %evl) :
         (vector<8xf32>, vector<8xf32>, vector<8xi1>, i32) -> vector<8xf32>
  vector.print %res : vector<8xf32>

  //===--------------------------------------------------------------------===//
  // Vector Config Operation + Add Operation + Fixed Vector Type
  //===--------------------------------------------------------------------===//

  %res_vector_config = bud.vector_config %mask, %evl : vector<8xi1>, i32 {
    %add = arith.addf %vec1, %vec2 : vector<8xf32>
    vector.yield %add : vector<8xf32>
  } : vector<8xf32>
  vector.print %res_vector_config : vector<8xf32>

  //===--------------------------------------------------------------------===//
  // VP Intrinsic Add Operation + Scalable Vector Type + RVV Dialect
  //===--------------------------------------------------------------------===//
  
  // Configure the register.
  // SEW = 32
  %sew = arith.constant 2 : index
  // LMUL = 2
  %lmul = arith.constant 1 : index
  // AVL = 8
  %avl = arith.constant 8 : index
  %vl = rvv.setvl %avl, %sew, %lmul : index
  %vl_i32 = arith.index_cast %vl : index to i32
  // Load vl elements and create the mask.
  %vec3 = rvv.load %mem[%c0], %vl : memref<20xf32>, vector<[4]xf32>, index
  %vec4 = rvv.load %mem[%c10], %vl : memref<20xf32>, vector<[4]xf32>, index
  %vec_mask = vector.create_mask %c6 : vector<[4]xi1>
  // Perform the fadd vp intrinsic.
  %res2 = "llvm.intr.vp.fadd" (%vec3, %vec4, %vec_mask, %vl_i32) :
         (vector<[4]xf32>, vector<[4]xf32>, vector<[4]xi1>, i32) -> vector<[4]xf32>
  // Print the result.
  %res2_mem = call @alloc_mem() : () -> memref<8xf32>
  rvv.store %res2, %res2_mem[%c0], %vl : vector<[4]xf32>, memref<8xf32>, index
  %print_res2 = memref.cast %res2_mem : memref<8xf32> to memref<*xf32>
  call @printMemrefF32(%print_res2) : (memref<*xf32>) -> ()

  %ret = arith.constant 0 : i32
  return %ret : i32
}
