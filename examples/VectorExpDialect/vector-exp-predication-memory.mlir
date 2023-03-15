memref.global "private" @gv_i32 : memref<2x10xi32> = dense<[[0 , 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9] ,
                                                            [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]]>

func.func private @printMemrefI32(memref<*xi32>)

func.func @main() -> i32 {
  %mem_i32 = memref.get_global @gv_i32 : memref<2x10xi32>
  %c0 = arith.constant 0 : index
  %mask = arith.constant dense<[1, 1, 1, 1, 1, 1, 0, 0]> : vector<8xi1>
  %vl = arith.constant 4 : i32
  
  %test = vector_exp.predication %mask, %vl : vector<8xi1>, i32 {
    %ele = vector.load %mem_i32[%c0, %c0] : memref<2x10xi32>, vector<8xi32>
    vector.yield %ele : vector<8xi32>
  } : vector<8xi32>

  vector_exp.predication %mask, %vl : vector<8xi1>, i32 {
    vector.store %test, %mem_i32[%c0, %c0] : memref<2x10xi32>, vector<8xi32>
    vector.yield
  } : () -> ()

  vector.print %test : vector<8xi32>

  %print_result_mem = memref.cast %mem_i32 : memref<2x10xi32> to memref<*xi32>
  call @printMemrefI32(%print_result_mem) : (memref<*xi32>) -> ()

  %ret = arith.constant 0 : i32
  return %ret : i32
}
