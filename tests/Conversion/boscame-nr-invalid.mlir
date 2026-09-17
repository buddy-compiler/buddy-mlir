// RUN: buddy-opt %s --lower-bosc-ame --split-input-file --verify-diagnostics

module attributes {bosc_ame.target = "nr-fpga"} {
  func.func @wrong_memory(%c: memref<16x16xf32>, %stride: i64) {
    // expected-error @+1 {{nr-fpga accumulator memory must be i32}}
    %acc = bosc_ame.mlce32.m %c, %stride {bosc_ame.fpga.slot = 0 : i32} : memref<16x16xf32> -> vector<4x4xi32>
    return
  }
}

// -----

module attributes {bosc_ame.target = "nr-fpga"} {
  func.func @unsupported_transpose(%b: memref<64x16xi8>, %stride: i64) {
    // expected-error @+1 {{nr-fpga does not support transposed B loads}}
    %tile = bosc_ame.mlbte8.m %b, %stride {bosc_ame.fpga.slot = 4 : i32} : memref<64x16xi8> -> vector<4x4xi8>
    return
  }
}

// -----

module attributes {bosc_ame.target = "nr-fpga"} {
  func.func @wrong_tile_memory(%b: memref<16x64xf32>, %stride: i64) {
    // expected-error @+1 {{nr-fpga A/B tile memory must be i8}}
    %tile = bosc_ame.mlbe8.m %b, %stride {bosc_ame.fpga.slot = 4 : i32} : memref<16x64xf32> -> vector<4x4xi8>
    return
  }
}

// -----

module attributes {bosc_ame.target = "nr-fpga"} {
  func.func @unsupported_rank_one(%b: memref<16x64xi32>, %stride: i64) {
    // expected-error @+1 {{is not supported by the FPGA prototype convention}}
    %tile = bosc_ame.mlae32.m %b, %stride : memref<16x64xi32> -> vector<16xi32>
    return
  }
}

// -----

module attributes {bosc_ame.target = "nr-fpga"} {
  func.func @upstream_intrinsic(%b: !llvm.ptr, %stride: i64) {
    // expected-error @+1 {{nr-fpga cannot use an upstream matrix intrinsic}}
    %tile = "bosc_ame.intr.mlae8.m"(%b, %stride) : (!llvm.ptr, i64) -> vector<[128]xi8>
    return
  }
}

// -----

module attributes {bosc_ame.target = "nr-fpga"} {
  func.func @immediate_config() {
    // expected-error @+1 {{nr-fpga requires register-form tile configuration}}
    %tile = bosc_ame.msettilemi 16 : i64
    return
  }
}

// -----

module attributes {bosc_ame.target = "nr-fpga"} {
  func.func @wrong_tile_type(%b: memref<16x64xi8>, %stride: i64) {
    // expected-error @+1 {{requires an 8-bit tile}}
    %tile = bosc_ame.mlbe8.m %b, %stride {bosc_ame.fpga.slot = 4 : i32} : memref<16x64xi8> -> vector<16xi32>
    return
  }
}
