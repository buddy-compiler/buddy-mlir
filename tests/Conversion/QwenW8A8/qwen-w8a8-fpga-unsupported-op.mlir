// RUN: not buddy-opt %s --lower-bosc-ame 2>&1 | FileCheck %s
//
// The FPGA convention maps register files by element width (see
// docs/BOSCAMEFPGAValueSemantics.md), so it only covers the W8A8 datapath: i8
// A/B tiles with an i32 accumulator.  An AME operation outside that datapath
// must be rejected with a diagnostic instead of silently selecting the wrong
// register file.  The same operations stay legal under the default (upstream)
// profile, which the BOSCAMEDialect examples cover.
//
// CHECK: error: BOSC AME operation 'bosc_ame.mlae32.m' is not supported by the FPGA prototype convention
module attributes {bosc_ame.target = "qwen3-fpga"} {
  func.func @i32_tile_load(%A: memref<4x4xi32>, %stride: i64) -> vector<4x4xi32> {
    %tile = bosc_ame.mlae32.m %A, %stride : memref<4x4xi32> -> vector<4x4xi32>
    return %tile : vector<4x4xi32>
  }
}
