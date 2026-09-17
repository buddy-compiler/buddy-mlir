// RUN: buddy-opt %s --split-input-file --lower-bosc-ame --verify-diagnostics
module attributes {bosc_ame.target = "qwen3-fpga"} {
  // expected-error @+1 {{target features disable the module's qwen3-fpga AME contract}}
  func.func @disabled() attributes {llvm.target_features = #llvm.target_features<["-xboscame-fpga"]>} { return }
}
// -----
module attributes {bosc_ame.target = "qwen3-fpga"} {
  // expected-error @+1 {{target features disable the module's qwen3-fpga AME contract}}
  llvm.func @disabled() attributes {target_features = #llvm.target_features<["-xboscame"]>} { llvm.return }
}
// -----
module attributes {bosc_ame.target = "qwen3-fpga"} {
  func.func @integer_store(%acc: vector<4x4xi32>, %out: memref<4x4xi32>, %stride: i64) {
    // expected-error @+1 {{qwen3-fpga accumulator memory must be f32}}
    bosc_ame.msce32.m %acc, %out, %stride : vector<4x4xi32>, memref<4x4xi32>
    return
  }
}
