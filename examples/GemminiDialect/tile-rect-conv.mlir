// RUN: buddy-opt %s \
// RUN:     --lower-gemmini | \
// RUN: FileCheck %s

// batchSize = 1 inputRowDim = 5 inputColDim = 10 inChannels = 1
memref.global "private" @input : memref<1x5x10x1xi8> = dense<[[[[1], [0], [-1], [0], [1], [1], [0], [-1], [0], [1]],
                                                              [[1], [0], [-1], [0], [1], [1], [0], [-1], [0], [1]],
                                                              [[1], [0], [-1], [0], [1], [1], [0], [-1], [0], [1]],
                                                              [[1], [0], [-1], [0], [1], [1], [0], [-1], [0], [1]],
                                                              [[1], [0], [-1], [0], [1], [1], [0], [-1], [0], [1]]]]>

// outChannels = 2 kernelDim = 3 inChannels = 1
memref.global "private" @weight : memref<9x2xi8> = dense<[[1, 2], [1, 2], [1, 2],
                                                          [1, 2], [1, 2], [1, 2],
                                                          [1, 2], [1, 2], [1, 2]]>

// outChannels = 2
memref.global "private" @bias : memref<2xi32> = dense<[1,1]>

func.func @main() -> i64 {
  %0 = arith.constant 0 : i64
  %3 = arith.constant 3 : i64
  %8 = arith.constant 8 : i64
  
  %input = memref.get_global @input : memref<1x5x10x1xi8>
  %weight = memref.get_global @weight : memref<9x2xi8>
  %bias = memref.get_global @bias : memref<2xi32>
  %output = memref.alloc() : memref<24x2xi8>
  // CHECK: "gemmini.intr.config_st"
  // CHECK: "gemmini.intr.config_ex"
  // CHECK: "gemmini.intr.config_ld"
  // CHECK: "gemmini.intr.mvin3"
  // CHECK: "gemmini.intr.mvin"
  // CHECK: "gemmini.intr.mvin2"
  // CHECK: "gemmini.intr.preload"
  // CHECK: "gemmini.intr.compute_preloaded"
  // CHECK: "gemmini.intr.compute_accumulated"
  gemmini.tile_conv %input %weight %bias %output %3 %8 %3 {stride = 1}:
  memref<1x5x10x1xi8> memref<9x2xi8> memref<2xi32> memref<24x2xi8> i64 i64 i64
  gemmini.print %output : memref<24x2xi8>
  return %0 : i64
}
