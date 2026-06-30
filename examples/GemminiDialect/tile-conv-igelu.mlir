// RUN: buddy-opt %s \
// RUN:     --lower-gemmini | \
// RUN: FileCheck %s
// RUN: buddy-opt %s \
// RUN:     --lower-gemmini | \
// RUN: buddy-translate -buddy-to-llvmir | \
// RUN: buddy-llc -filetype=asm -mtriple=riscv64 \
// RUN:     -mattr=+xgemmini,+D -float-abi=hard \
// RUN:     -o - | FileCheck %s --check-prefix=ASM

// batchSize = 1 inputDim = 5 inChannels = 1
memref.global "private" @input : memref<1x5x5x1xi8> = dense<[[[[1], [0], [-1], [0], [1]],
                                                              [[1], [0], [-1], [0], [1]],
                                                              [[1], [0], [-1], [0], [1]],
                                                              [[1], [0], [-1], [0], [1]],
                                                              [[1], [0], [-1], [0], [1]]]]>

// outChannels = 2 kernelDim = 3 inChannels = 1
memref.global "private" @weight : memref<9x2xi8> = dense<[[-1, 2], [-1, 2], [-1, 2],
                                                          [-1, 2], [-1, 2], [-1, 2],
                                                          [-1, 2], [-1, 2], [-1, 2]]>

// outChannels = 2
memref.global "private" @bias : memref<2xi32> = dense<[1,1]>

func.func @main() -> i64 {
  %0 = arith.constant 0 : i64
  %3 = arith.constant 3 : i64
  %input = memref.get_global @input : memref<1x5x5x1xi8>
  %weight = memref.get_global @weight : memref<9x2xi8>
  %bias = memref.get_global @bias : memref<2xi32>
  %output = memref.alloc() : memref<9x2xi8>

  // CHECK: "gemmini.intr.loop_conv_ws_config1"
  // CHECK: "gemmini.intr.loop_conv_ws_config2"
  // CHECK: "gemmini.intr.loop_conv_ws_config3"
  // CHECK: "gemmini.intr.loop_conv_ws_config4"
  // CHECK: "gemmini.intr.loop_conv_ws_config5"
  // CHECK: "gemmini.intr.loop_conv_ws_config6"
  // CHECK: "gemmini.intr.loop_conv_ws"
  // CHECK: "gemmini.intr.flush"
  gemmini.tile_conv %input %weight %bias %output %3 %3 %3 {stride = 1}:
  memref<1x5x5x1xi8> memref<9x2xi8> memref<2xi32> memref<9x2xi8> i64 i64 i64
  gemmini.print %output : memref<9x2xi8>

  // CHECK: "gemmini.intr.loop_conv_ws_config1"
  // CHECK: "gemmini.intr.loop_conv_ws_config2"
  // CHECK: "gemmini.intr.loop_conv_ws_config3"
  // CHECK: "gemmini.intr.loop_conv_ws_config4"
  // CHECK: "gemmini.intr.loop_conv_ws_config5"
  // CHECK: "gemmini.intr.loop_conv_ws_config6"
  // CHECK: "gemmini.intr.loop_conv_ws"
  // CHECK: "gemmini.intr.flush"
  gemmini.tile_conv %input %weight %bias %output %3 %3 %3 {stride = 1, act = 3}:
  memref<1x5x5x1xi8> memref<9x2xi8> memref<2xi32> memref<9x2xi8> i64 i64 i64
  gemmini.print %output : memref<9x2xi8>
  return %0 : i64
}

// ASM: .attribute 5, "{{.*xgemmini.*}}"
// ASM: loop_conv_ws_config1{{[ \t]}}
// ASM: loop_conv_ws{{[ \t]}}
// ASM: flush{{[ \t]}}
