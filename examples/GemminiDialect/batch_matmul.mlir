// RUN: buddy-opt %s \
// RUN:     --convert-linalg-to-gemmini | \
// RUN: FileCheck %s

func.func @main() -> i8 {
  %0 = arith.constant 0 : i8 
  %1 = arith.constant 1 : i8
  %2 = arith.constant 2 : i8 
  %input0 = memref.alloc() : memref<3x3x3xi8> 
  %input1 = memref.alloc() : memref<3x3x3xi8> 
  %output = memref.alloc() : memref<3x3x3xi8>  
  linalg.fill
    ins(%1 : i8)
  outs(%input0 : memref<3x3x3xi8>)
  linalg.fill
    ins(%2 : i8)
  outs(%input1 : memref<3x3x3xi8>)
  // CHECK: gemmini.tile_matmul %subview %subview_2 %subview_3 %alloc_4 : 
  // CHECK-SAME: memref<3x3xi8, strided<[3, 1]>> memref<3x3xi8, strided<[3, 1]>> memref<3x3xi8, strided<[3, 1]>> memref<3x3xi32>
  // CHECK: gemmini.tile_matmul %subview_5 %subview_6 %subview_7 %alloc_8 : 
  // CHECK-SAME: memref<3x3xi8, strided<[3, 1], offset: 9>> memref<3x3xi8, strided<[3, 1], offset: 9>> memref<3x3xi8, strided<[3, 1], offset: 9>> memref<3x3xi32>
  // CHECK: gemmini.tile_matmul %subview_10 %subview_11 %subview_12 %alloc_13 : 
  // CHECK-SAME: memref<3x3xi8, strided<[3, 1], offset: 18>> memref<3x3xi8, strided<[3, 1], offset: 18>> memref<3x3xi8, strided<[3, 1], offset: 18>> memref<3x3xi32>
  linalg.batch_matmul
    ins(%input0, %input1: memref<3x3x3xi8>, memref<3x3x3xi8>)
  outs(%output : memref<3x3x3xi8>)
  gemmini.print %output : memref<3x3x3xi8>
  memref.dealloc %output : memref<3x3x3xi8> 
  return %0 : i8
}
