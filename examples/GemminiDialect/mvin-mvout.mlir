memref.global "private" @gv : memref<2x16xi8> = dense<[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                                                       [17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]]>

func.func @main() -> i64{
  %0 = arith.constant 0 : i64
  %stride = arith.constant 16 : i64
  %spadAddr = arith.constant 0 : i64
  %arrayA = memref.get_global @gv : memref<2x16xi8>
  %arrayB = memref.alloc() : memref<2x16xi8>
  gemmini.print %arrayA : memref<2x16xi8>
  %arrayAIndexAddr = memref.extract_aligned_pointer_as_index %arrayA : memref<2x16xi8> -> index 
  %arrayAI64Addr = arith.index_cast %arrayAIndexAddr : index to i64
  %arrayBIndexAddr = memref.extract_aligned_pointer_as_index %arrayB : memref<2x16xi8> -> index 
  %arrayBI64Addr = arith.index_cast %arrayBIndexAddr : index to i64 
  gemmini.configSt %stride : i64 
  gemmini.configLd %stride : i64 
  gemmini.mvin %arrayAI64Addr %spadAddr : i64 i64
  gemmini.mvout %arrayBI64Addr %spadAddr : i64 i64
  gemmini.print %arrayB : memref<2x16xi8>
  return %0 : i64
}
