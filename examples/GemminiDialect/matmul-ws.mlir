memref.global "private" @gv1 : memref<4x4xi8> = dense<[[1, 2, 3, 4],
                                                       [5, 6, 7, 8],
                                                       [9, 10, 11, 12],
                                                       [13, 14, 15, 16]]>
memref.global "private" @gv2 : memref<4x4xi8> = dense<[[1, 1, 1, 1],
                                                       [1, 1, 1, 1],
                                                       [1, 1, 1, 1],
                                                       [1, 1, 1, 1]]>
memref.global "private" @gv3 : memref<4x4xi8> = dense<[[2, 2, 2, 2],
                                                      [2, 2, 2, 2],
                                                      [2, 2, 2, 2],
                                                      [2, 2, 2, 2]]>

func.func @main() -> i64 {
  %aArray = memref.get_global @gv1 : memref<4x4xi8>
  %bArray = memref.get_global @gv2 : memref<4x4xi8>
  %cArray = memref.alloc() : memref<4x4xi8>
  %dArray = memref.get_global @gv3 : memref<4x4xi8>
  gemmini.print %cArray : memref<4x4xi8> 
  %aSpAddr = arith.constant 0 : i64
  %bSpAddr = arith.constant 4 : i64 
  %cSpAddr = arith.constant 8 : i64
  %dSpAddr = arith.constant 12 : i64 
  %cst4 = arith.constant 4 : i64
  %cst0 = arith.constant 0 : i64
  gemmini.config_st %cst4 : i64
  gemmini.config_ld %cst4 : i64
  gemmini.mvin %aArray %aSpAddr : memref<4x4xi8> i64
  gemmini.mvin %bArray %bSpAddr : memref<4x4xi8> i64
  gemmini.mvin %cArray %cSpAddr : memref<4x4xi8> i64 
  gemmini.mvin %dArray %dSpAddr : memref<4x4xi8> i64 
  gemmini.config_ex {dataflow = 1}
  gemmini.preload %bSpAddr %cSpAddr %cst4 %cst4 %cst4 %cst4: i64 i64 i64 i64 i64 i64
  gemmini.compute_preloaded %aSpAddr %dSpAddr %cst4 %cst4 %cst4 %cst4 : i64 i64 i64 i64 i64 i64
  gemmini.mvout %cArray %cSpAddr : memref<4x4xi8> i64
  gemmini.print %cArray : memref<4x4xi8> 
  return %cst0 : i64
}
