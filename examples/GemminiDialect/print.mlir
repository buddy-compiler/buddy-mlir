// RUN: buddy-opt %s \
// RUN:     --convert-linalg-to-gemmini \
//          --convert-linalg-to-loops \
//          --lower-gemmini | \
// RUN: FileCheck %s

func.func @main() -> i8 {
  %c0 = arith.constant 0 : i8

  %scalar = arith.constant 42 : i8
  // CHECK: gemmini.print_scalar %{{.*}} : i8
  gemmini.print_scalar %scalar : i8

  %vector = memref.alloc() : memref<4xi8>        // 1D向量
  %matrix = memref.alloc() : memref<2x3xi8>      // 2D矩阵
  %tensor = memref.alloc() : memref<1x2x3xi8>    // 3D张量
  %c1 = arith.constant 1 : i8
  linalg.fill ins(%c1 : i8) outs(%vector : memref<4xi8>)
  linalg.fill ins(%c1 : i8) outs(%matrix : memref<2x3xi8>)
  // CHECK: gemmini.print %{{.*}} : memref<4xi8>
  gemmini.print %vector : memref<4xi8>
  // CHECK: gemmini.print %{{.*}} : memref<2x3xi8>
  gemmini.print %matrix : memref<2x3xi8>
  // CHECK: gemmini.print %{{.*}} : memref<1x2x3xi8>
  gemmini.print %tensor : memref<1x2x3xi8>
  memref.dealloc %vector : memref<4xi8>
  memref.dealloc %matrix : memref<2x3xi8>
  memref.dealloc %tensor : memref<1x2x3xi8>

  return %c0 : i8
}
