// RUN: buddy-opt %s \
// RUN:     --convert-linalg-to-gemmini="acc_t=f32" | \
// RUN: FileCheck %s

memref.global "private" @input : memref<1x5x5x1xf32> = dense<[[[[1.],[2.],[3.],[4.],[5.]],
                                                               [[6.],[7.],[8.],[9.],[10.]],
                                                               [[11.],[12.],[13.],[14.],[15.]],
                                                               [[16.],[17.],[18.],[19.],[20.]],
                                                               [[21.],[22.],[23.],[24.],[25.]]]]>

memref.global "private" @kernel : memref<1x3x3x1xf32> = dense<[[[[1.], [1.], [1.]],
                                                                [[1.], [1.], [1.]],
                                                                [[1.], [1.], [1.]]]]>


func.func @main() -> i8 {
  %0 = arith.constant 0 : i8
  // batchsize = 2 inputchannel = 2
  %input = memref.get_global @input : memref<1x5x5x1xf32>
  // outputchannel = 3
  %kernel = memref.get_global @kernel : memref<1x3x3x1xf32>
  // batchsize h w outputchannel
  %output = memref.alloc() : memref<1x3x3x1xf32>
  // CHECK: gemmini.tile_conv %{{.+}} %alloc_{{[0-9]+}} %alloc_{{[0-9]+}} %alloc_{{[0-9]+}} %{{.+}} %{{.+}} :
  // CHECK: memref<1x5x5x1xf32> memref<9x1xf32> memref<1xf32> memref<9x1xf32> i64 i64
  linalg.conv_2d_nhwc_fhwc
    ins(%input, %kernel : memref<1x5x5x1xf32>, memref<1x3x3x1xf32>)
  outs(%output : memref<1x3x3x1xf32>)
  gemmini.print %output : memref<1x3x3x1xf32>
  return %0 : i8
}
