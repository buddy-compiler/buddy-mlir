memref.global "private" @gv1 : memref<4x4xi8> = dense<[[1, 2, 3, 4],
                                                       [5, 6, 7, 8],
                                                       [9, 10, 11, 12],
                                                       [13, 14, 15, 16]]>
memref.global "private" @gv2 : memref<4x4xi8> = dense<[[1, 0, 0, 0],
                                                       [0, 1, 0, 0],
                                                       [0, 0, 1, 0],
                                                       [0, 0, 0, 1]]>

func.func @main() -> i64 {
  %in = memref.get_global @gv1 : memref<4x4xi8>
  %identity = memref.get_global @gv2 : memref<4x4xi8>
  %out = memref.alloc() : memref<4x4xi8>
  gemmini.print %out : memref<4x4xi8> 
  %inSpAddr = arith.constant 0 : i64
  %outSpAddr = arith.constant 4 : i64 
  %identitySpAddr = arith.constant 8 : i64
  %cst4 = arith.constant 4 : i64
  %cst0 = arith.constant 0 : i64
  gemmini.configSt %cst4 : i64
  gemmini.configLd %cst4 : i64
  gemmini.mvin %in %inSpAddr : memref<4x4xi8> i64
  gemmini.configLd %cst4 : i64
  gemmini.mvin %identity %identitySpAddr : memref<4x4xi8> i64
  gemmini.configEx {dataflow = 0, aTranspose = true } 
  gemmini.preloadZeros %outSpAddr %cst4 %cst4 : i64 i64 i64
  gemmini.computePreloaded %inSpAddr %identitySpAddr %cst4 %cst4 %cst4 %cst4 : i64 i64 i64 i64 i64 i64
  gemmini.mvout %out %outSpAddr : memref<4x4xi8> i64
  gemmini.print %out : memref<4x4xi8> 
  return %cst0 : i64
}
