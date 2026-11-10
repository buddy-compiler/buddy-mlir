// RUN: buddy-opt %s --lower-bosc-ame | FileCheck %s
//
// Private DDR frame acquire/release lowers to runtime ABI calls. Views into
// the frame keep the shifted base so later DMA/tile uses see aligned storage.
//
module attributes {bosc_ame.target = "nr-fpga", bosc_ame.cpu_ddr_policy} {
  // CHECK-LABEL: llvm.func @private_accumulator
  func.func @private_accumulator() {
    // CHECK: llvm.call @bosc_ame_private_ddr_acquire
    %frame = bosc_ame.ddr.frame : memref<512xi8>
    %shift = arith.constant 64 : index
    %view = memref.view %frame[%shift][] : memref<512xi8> to memref<2x16xi32>
    // CHECK: llvm.call_intrinsic "llvm.riscv.bosc.ordered.load.acc"
    %acc = bosc_ame.tile.load_acc %view {bosc_ame.direct_ddr} : memref<2x16xi32> -> vector<2x16xi32>
    // CHECK: llvm.call_intrinsic "llvm.riscv.bosc.ordered.store.acc"
    bosc_ame.tile.store_acc %acc, %view {bosc_ame.direct_ddr} : vector<2x16xi32>, memref<2x16xi32>
    // CHECK: llvm.call @bosc_ame_private_ddr_release
    bosc_ame.ddr.release %frame : memref<512xi8>
    return
  }
}
