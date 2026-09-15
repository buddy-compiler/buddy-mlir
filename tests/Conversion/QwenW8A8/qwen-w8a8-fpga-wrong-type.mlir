// RUN: not buddy-opt %s --lower-bosc-ame 2>&1 | FileCheck %s
//
// Companion to qwen-w8a8-fpga-unsupported-op.mlir: this operation *is* part of
// the FPGA W8A8 schedule, but with the wrong element type.  A tile load must
// carry an 8-bit tile, so an i32 result has to be diagnosed rather than mapped
// to the accumulator register file by the width heuristic.
//
// CHECK: error: BOSC AME operation 'bosc_ame.mlae8.m' uses 'i32' where the FPGA prototype convention requires an 8-bit tile
module attributes {bosc_ame.target = "qwen3-fpga"} {
  func.func @i32_tile_as_a_load(%A: memref<4x4xi32>, %stride: i64) -> vector<4x4xi32> {
    %tile = bosc_ame.mlae8.m %A, %stride : memref<4x4xi32> -> vector<4x4xi32>
    return %tile : vector<4x4xi32>
  }
}
