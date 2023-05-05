func.func @main() -> i8 {
  %0 = arith.constant 0 : i8 
  %1 = arith.constant 1 : i8
  %2 = arith.constant 2 : i8 
  %input0 = memref.alloc() { alignment = 16 } : memref<3x3x3xi8> 
  %input1 = memref.alloc() { alignment = 16 } : memref<3x3x3xi8> 
  %output = memref.alloc() { alignment = 16 } : memref<3x3x3xi8>  
  linalg.fill
    ins(%1 : i8)
  outs(%input0 : memref<3x3x3xi8>)
  linalg.fill
    ins(%2 : i8)
  outs(%input1 : memref<3x3x3xi8>)
  linalg.batch_matmul
    ins(%input0, %input1: memref<3x3x3xi8>, memref<3x3x3xi8>)
  outs(%output : memref<3x3x3xi8>)
  gemmini.print %output : memref<3x3x3xi8>
  memref.dealloc %output : memref<3x3x3xi8> 
  return %0 : i8
}
