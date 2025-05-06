// RUN: buddy-opt %s \
// RUN:     --convert-linalg-to-gemmini | \
// RUN: FileCheck %s

func.func @main() -> i8 {
  %0 = arith.constant 0 : i8
  %1 = arith.constant 1 : i8
  %2 = arith.constant 2 : i8
  %mem0 = memref.alloc() : memref<32x32xi8>
  %mem1 = memref.alloc() : memref<32x32xi8>
  %mem2 = memref.alloc() : memref<32x32xi8>
  linalg.fill
    ins(%2 : i8)
  outs(%mem0 : memref<32x32xi8>)
  linalg.fill
    ins(%1 : i8)
  outs(%mem1 : memref<32x32xi8>)
  linalg.matmul
    ins(%mem0, %mem1 : memref<32x32xi8>, memref<32x32xi8>)
  outs(%mem2 : memref<32x32xi8>)
  gemmini.print %mem2 : memref<32x32xi8>
  memref.dealloc %mem0 : memref<32x32xi8>
  memref.dealloc %mem1 : memref<32x32xi8>
  memref.dealloc %mem2 : memref<32x32xi8>
  return %0 : i8
}
