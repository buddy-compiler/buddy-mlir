// RUN: buddy-opt %s \
// RUN:     --convert-linalg-to-gemmini | \
// RUN: FileCheck %s

memref.global "private" @input : memref<1x7x7x1xi8> = dense<[[[[1],[1],[1],[1],[1],[1],[1]],
                                                              [[1],[1],[1],[1],[1],[1],[1]],
                                                              [[1],[1],[1],[1],[1],[1],[1]],
                                                              [[1],[1],[1],[1],[1],[1],[1]],
                                                              [[1],[1],[1],[1],[1],[1],[1]],
                                                              [[1],[1],[1],[1],[1],[1],[1]],
                                                              [[1],[1],[1],[1],[1],[1],[1]]]]>

memref.global "private" @kernel : memref<1x5x5x1xi8> = dense<[[[[1], [1], [1], [1], [1]],
                                                               [[1], [1], [1], [1], [1]],
                                                               [[1], [1], [1], [1], [1]],
                                                               [[1], [1], [1], [1], [1]],
                                                               [[1], [1], [1], [1], [1]]]]>

func.func @main() -> i8 {
  %0 = arith.constant 0 : i8
  %input = memref.get_global @input : memref<1x7x7x1xi8>
  %kernel = memref.get_global @kernel : memref<1x5x5x1xi8>
  %output = memref.alloc() : memref<1x3x3x1xi8>

  // CHECK: gemmini.tile_conv %{{[0-9]+}} %alloc_{{[0-9]+}} %alloc_{{[0-9]+}} %alloc_{{[0-9]+}} %{{.+}} %{{.+}} :
  // CHECK-SAME: memref<1x7x7x1xi8> memref<25x1xi8> memref<1xi32> memref<9x1xi8> i64 i64
  linalg.conv_2d_nhwc_fhwc
    ins(%input, %kernel : memref<1x7x7x1xi8>, memref<1x5x5x1xi8>)
  outs(%output : memref<1x3x3x1xi8>)
  gemmini.print %output : memref<1x3x3x1xi8>
  return %0 : i8
}
