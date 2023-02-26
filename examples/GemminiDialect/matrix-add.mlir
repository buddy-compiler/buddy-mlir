memref.global "private" @gv1 : memref<4x4xi8> = dense<[[1, 2, 3, 4],
                                                       [5, 6, 7, 8],
                                                       [9, 10, 11, 12],
                                                       [13, 14, 15, 16]]>
memref.global "private" @gv2 : memref<4x4xi8> = dense<[[17, 18, 19, 20],
                                                       [21, 22, 23, 24],
                                                       [25, 26, 27, 28],
                                                       [29, 30, 31, 32]]>

func.func @main() -> i64 {
  %arrayA = memref.get_global @gv1 : memref<4x4xi8>
  %arrayB = memref.get_global @gv2 : memref<4x4xi8>
  %arrayC = memref.alloc() : memref<4x4xi8>
  gemmini.print %arrayC : memref<4x4xi8>
  // 10000000000000000000000000000000  
  %a_Acc_Addr = arith.constant 2147483648 : i64
  // 11000000000000000000000000000000
  %b_Acc_Addr = arith.constant 3221225472 : i64 
  // 10000000000000000000000000000000
  %c_Acc_Addr = arith.constant 2147483648
  %cst4 = arith.constant 4 : i64
  %cst0 = arith.constant 0 : i64
  gemmini.configLd %cst4 {shrunk = true} : i64
  gemmini.mvin %arrayA %a_Acc_Addr : memref<4x4xi8> i64
  gemmini.configLd %cst4 {shrunk = true} : i64
  gemmini.mvin %arrayB %b_Acc_Addr : memref<4x4xi8> i64
  gemmini.configEx {dataflow = 0, sys_act = 0, sys_shift = 0}
  gemmini.configSt %cst4 : i64
  gemmini.mvout %arrayC %c_Acc_Addr : memref<4x4xi8> i64
  gemmini.print %arrayC : memref<4x4xi8> 
  return %cst0 : i64
}
