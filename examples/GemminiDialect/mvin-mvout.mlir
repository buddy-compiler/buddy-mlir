memref.global "private" @gv : memref<2x16xi8> = dense<[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                                                       [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]]>

func.func @main() -> i64{
  %0 = arith.constant 0 : i64
  %stride = arith.constant 16 : i64
  %spadAddr = arith.constant 0 : i64
  %arrayA = memref.get_global @gv : memref<2x16xi8>
  %arrayB = memref.alloc() : memref<2x16xi8>
  gemmini.print %arrayB : memref<2x16xi8>
  gemmini.configSt %stride : i64 
  gemmini.configLd %stride : i64 
  gemmini.mvin %arrayA %spadAddr : memref<2x16xi8> i64
  gemmini.mvout %arrayB %spadAddr : memref<2x16xi8> i64
  gemmini.print %arrayB : memref<2x16xi8>
  return %0 : i64
}
