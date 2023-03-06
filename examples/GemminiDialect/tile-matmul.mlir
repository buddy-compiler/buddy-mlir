func.func @main() -> i8 {
  %i0 = arith.constant 0 : i8
  %i1I8 = arith.constant 1 : i8
  %i1I32 = arith.constant 1 : i32
  %c0 = arith.constant 0 : index 
  %c1 = arith.constant 1 : index 
  %c4 = arith.constant 4 : index 
  %aArray = memref.alloc() {alignment = 16} : memref<4x4xi8>
  %bArray = memref.alloc() {alignment = 16} : memref<4x4xi8>
  %cArray = memref.alloc() {alignment = 16} : memref<4x4xi8>
  %dArray = memref.alloc() {alignment = 64} : memref<4x4xi32>
  %dim = memref.dim %aArray, %c0 : memref<4x4xi8> 
  scf.for %i = %c0 to %dim step %c1 {
    scf.for %j = %c0 to %dim step %c1 {
      %temp0 = arith.muli %i, %c4: index
      %temp1 = arith.addi %temp0, %j : index 
      %cast = arith.index_cast %temp1 : index to i8
      memref.store %cast, %aArray[%i, %j] : memref<4x4xi8> 
      memref.store %i1I8, %bArray[%i, %j] : memref<4x4xi8> 
      memref.store %i1I32, %dArray[%i, %j] : memref<4x4xi32>
    }
  }
  gemmini.print %aArray : memref<4x4xi8>
  gemmini.print %bArray : memref<4x4xi8>
  gemmini.print %dArray : memref<4x4xi32>
  gemmini.tileMatMul %aArray %bArray %cArray %dArray : memref<4x4xi8> memref<4x4xi8> memref<4x4xi8> memref <4x4xi32>
  gemmini.print %cArray : memref<4x4xi8>
  return %i0 : i8
}
