memref.global "private" @gv : memref<4x10xf32> = dense<[[0. , 1. , 2. , 3. , 4. , 5. , 6. , 7. , 8. , 9. ],
                                                       [10., 11., 12., 13., 14., 15., 16., 17., 18., 19.],
                                                       [20., 21., 22., 23., 24., 25., 26., 27., 28., 29.],
                                                       [30., 31., 32., 33., 34., 35., 36., 37., 38., 39.]]>

func.func private @printMemrefF32(memref<*xf32>)

func.func @memref_add_offset(%arg0 : memref<?x?xf32>, %arg1 : f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %f0 = arith.constant 0.0 : f32
  %row = memref.dim %arg0, %c0 : memref<?x?xf32>
  %col = memref.dim %arg0, %c1 : memref<?x?xf32>
  %passthr_vec = vector.splat %f0 : vector<4xf32>
  %offset_vec = vector.splat %arg1 : vector<4xf32>
  scf.for %arg2 = %c0 to %row step %c1 {
    scf.for %arg3 = %c0 to %col step %c4 {
      %tail_cond = arith.subi %col, %arg3 : index
      %is_not_tail = arith.cmpi sge, %tail_cond, %c4 : index
      scf.if %is_not_tail {
        %mem_vec = vector.load %arg0[%arg2, %arg3] : memref<?x?xf32>, vector<4xf32>
        %result_vec = arith.addf %mem_vec, %offset_vec : vector<4xf32>
        vector.store %result_vec, %arg0[%arg2, %arg3] : memref<?x?xf32>, vector<4xf32>
      } else {
        %mask = vector.create_mask %tail_cond : vector<4xi1>
        %mem_vec = vector.maskedload %arg0[%arg2, %arg3], %mask, %passthr_vec : memref<?x?xf32>, vector<4xi1>, vector<4xf32> into vector<4xf32>
        %result_vec = arith.addf %mem_vec, %offset_vec : vector<4xf32>
        vector.maskedstore %arg0[%arg2, %arg3], %mask, %result_vec : memref<?x?xf32>, vector<4xi1>, vector<4xf32>
      }
    }
  }
  return
}

func.func @main() -> i32 {
  %mem_global = memref.get_global @gv : memref<4x10xf32>
  %mem = memref.cast %mem_global : memref<4x10xf32> to memref<?x?xf32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %offset = arith.constant 1.0 : f32

  call @memref_add_offset(%mem, %offset) : (memref<?x?xf32>, f32) -> ()

  %print_mem = memref.cast %mem : memref<?x?xf32> to memref<*xf32>
  call @printMemrefF32(%print_mem) : (memref<*xf32>) -> ()

  %ret = arith.constant 0 : i32
  return %ret : i32
}
