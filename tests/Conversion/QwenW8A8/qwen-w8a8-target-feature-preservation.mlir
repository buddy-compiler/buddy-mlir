// RUN: buddy-opt %s --split-input-file --lower-bosc-ame --convert-func-to-llvm | FileCheck %s
module attributes {bosc_ame.target = "qwen3-fpga"} {
  func.func @from_func() attributes {llvm.target_features = #llvm.target_features<["+m", "+v"]>} { return }
}
// CHECK: llvm.func @from_func() attributes {target_features = #llvm.target_features<["+xboscame-fpga", "+m", "+v"]>}
// -----
module attributes {bosc_ame.target = "qwen3-fpga"} {
  llvm.func @from_llvm() attributes {target_features = #llvm.target_features<["+m", "+v"]>} { llvm.return }
}
// CHECK: llvm.func @from_llvm() attributes {target_features = #llvm.target_features<["+xboscame-fpga", "+m", "+v"]>}
// -----
module {
  func.func @upstream() attributes {llvm.target_features = #llvm.target_features<["+m", "+v"]>} { return }
}
// CHECK: llvm.func @upstream() attributes {target_features = #llvm.target_features<["+m", "+v"]>}
