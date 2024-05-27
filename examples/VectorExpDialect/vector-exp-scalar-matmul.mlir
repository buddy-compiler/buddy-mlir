memref.global "private" @gv_i32 : memref<10x10xi32> = dense<[[0 , 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9],
                                                             [0 , 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9],
                                                             [0 , 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9],
                                                             [0 , 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9],
                                                             [0 , 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9],
                                                             [0 , 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9],
                                                             [0 , 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9],
                                                             [0 , 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9],
                                                             [0 , 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9],
                                                             [0 , 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9]]>

func.func private @printMemrefI32(memref<*xi32>)

func.func @alloc_mem_i32() -> memref<10x10xi32> {
  %i0 = arith.constant 0 : i32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %mem = memref.alloc() : memref<10x10xi32>
  scf.for %idx0 = %c0 to %c10 step %c1 {
    scf.for %idx1 = %c0 to %c10 step %c1 {
      memref.store %i0, %mem[%idx0, %idx1] : memref<10x10xi32>
    }
  }
  return %mem : memref<10x10xi32>
}

func.func @matmul(%a : memref<10x10xi32>, %b : memref<10x10xi32>, %c : memref<10x10xi32>) {
  linalg.matmul 
    ins(%a, %b: memref<10x10xi32>, memref<10x10xi32>)
    outs(%c:memref<10x10xi32>)
  return
}

func.func @main() -> i32 {
  %mem_i32 = memref.get_global @gv_i32 : memref<10x10xi32>
  %result_mem = call @alloc_mem_i32() : () -> memref<10x10xi32>

  call @matmul(%mem_i32, %mem_i32, %result_mem) : (memref<10x10xi32>, memref<10x10xi32>, memref<10x10xi32>) -> ()

  // %print_result_mem = memref.cast %result_mem : memref<10x10xi32> to memref<*xi32>
  // call @printMemrefI32(%print_result_mem) : (memref<*xi32>) -> ()

  %ret = arith.constant 0 : i32
  return %ret : i32
}
