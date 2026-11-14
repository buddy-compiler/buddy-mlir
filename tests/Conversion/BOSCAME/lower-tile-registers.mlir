// RUN: buddy-opt %s --lower-bosc-ame-tile | FileCheck %s

// MLIR does not allocate AME registers.
//
// The tile operations name no register and the lowering emits the ordinary
// ordered LLVM target intrinsics; which tile (`tr0`..`tr7`) and accumulator
// (`acc0`..`acc7`) register holds a value is decided by LLVM's register
// allocator.  This test pins the *instruction* sequence and the operands the
// lowering does own: two A tiles that are live at the same time - the case a
// fixed-slot convention would have to reject - lower to two independent loads,
// and placing them is the backend's job.
//
// CHECK-LABEL: llvm.func @simultaneous_tiles
// CHECK: llvm.riscv.bosc.ordered.load.a
// CHECK: llvm.riscv.bosc.ordered.load.a
// CHECK: llvm.riscv.bosc.ordered.load.b
// CHECK: llvm.riscv.bosc.ordered.load.acc
// CHECK: llvm.riscv.bosc.ordered.mma
// CHECK: llvm.riscv.bosc.ordered.mma
// CHECK: llvm.riscv.bosc.ordered.store.acc
// CHECK-NOT: bosc_ame.intr.fpga

module attributes {bosc_ame.target = "nr-fpga"} {
  func.func @simultaneous_tiles() -> i32 {
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %buf = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64} : memref<4096xi8, #bosc_ame.spm<0>>
    %a0 = memref.reinterpret_cast %buf to offset: [0], sizes: [8, 16], strides: [16, 1] : memref<4096xi8, #bosc_ame.spm<0>> to memref<8x16xi8, strided<[16, 1], offset: 0>, #bosc_ame.spm<0>>
    %a1 = memref.reinterpret_cast %buf to offset: [128], sizes: [8, 16], strides: [16, 1] : memref<4096xi8, #bosc_ame.spm<0>> to memref<8x16xi8, strided<[16, 1], offset: 128>, #bosc_ame.spm<0>>
    %b_buf = bosc_ame.spm.view {bank = 1 : i32, offset = 0 : i64} : memref<512xi8, #bosc_ame.spm<1>>
    %b = memref.reinterpret_cast %b_buf to offset: [0], sizes: [16, 32], strides: [1, 16] : memref<512xi8, #bosc_ame.spm<1>> to memref<16x32xi8, strided<[1, 16], offset: 0>, #bosc_ame.spm<1>>
    %acc = memref.view %buf[%c0][] : memref<4096xi8, #bosc_ame.spm<0>> to memref<8x32xi32, #bosc_ame.spm<0>>

    // Both A tiles are loaded before either one is consumed.
    %ta0 = bosc_ame.tile.load_a %a0 : memref<8x16xi8, strided<[16, 1], offset: 0>, #bosc_ame.spm<0>> -> vector<8x16xi8>
    %ta1 = bosc_ame.tile.load_a %a1 : memref<8x16xi8, strided<[16, 1], offset: 128>, #bosc_ame.spm<0>> -> vector<8x16xi8>
    %tb = bosc_ame.tile.load_b %b : memref<16x32xi8, strided<[1, 16], offset: 0>, #bosc_ame.spm<1>> -> vector<16x32xi8>
    %tc = bosc_ame.tile.load_acc %acc : memref<8x32xi32, #bosc_ame.spm<0>> -> vector<8x32xi32>

    %c1 = bosc_ame.tile.mma %tc, %ta0, %tb : vector<8x32xi32>, vector<8x16xi8>, vector<16x32xi8> -> vector<8x32xi32>
    %c2 = bosc_ame.tile.mma %c1, %ta1, %tb : vector<8x32xi32>, vector<8x16xi8>, vector<16x32xi8> -> vector<8x32xi32>
    bosc_ame.tile.store_acc %c2, %acc : vector<8x32xi32>, memref<8x32xi32, #bosc_ame.spm<0>>
    return %c0_i32 : i32
  }
}
