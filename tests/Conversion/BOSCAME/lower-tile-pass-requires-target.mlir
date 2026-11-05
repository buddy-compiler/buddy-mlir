// RUN: not buddy-opt %s --lower-bosc-ame-tile 2>&1 | FileCheck %s --check-prefix=ERR
// RUN: buddy-opt %s --lower-bosc-ame | FileCheck %s

// `--lower-bosc-ame-tile` is the FPGA tile entry point: it requires the module
// to select a contract this build implements, even when the tile operations are
// not present yet (they may be introduced earlier in the pipeline).
//
// `--lower-bosc-ame` on the same module is the upstream path and must keep
// working: a module without tile operations is lowered exactly as before, with
// no target feature and no FPGA intrinsic.

// ERR: does not select an AME hardware contract

// CHECK-LABEL: llvm.func @plain
// CHECK-NOT: target_features
// CHECK: llvm.return
func.func @plain(%x: i32) -> i32 {
  return %x : i32
}
