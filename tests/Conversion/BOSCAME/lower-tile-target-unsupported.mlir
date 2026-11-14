// RUN: not buddy-opt %s --lower-bosc-ame-tile 2>&1 | FileCheck %s --check-prefix=ERR

// A contract this build does not implement is rejected by name instead of
// being lowered with the wrong register file or accumulator semantics.

// ERR: bosc_ame.target = "qwen3-fpga"
// ERR-SAME: has no verified tile datapath in this version
// ERR-SAME: "nr-fpga"

module attributes {bosc_ame.target = "qwen3-fpga"} {
  func.func @unsupported_target(%a: memref<4096xi8>) -> i32 {
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %buf = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64} : memref<4096xi8, #bosc_ame.spm<0>>
    %tile = memref.view %buf[%c0][] : memref<4096xi8, #bosc_ame.spm<0>> to memref<64x64xi8, #bosc_ame.spm<0>>
    %ta = bosc_ame.tile.load_a %tile : memref<64x64xi8, #bosc_ame.spm<0>> -> vector<64x64xi8>
    %acc_buf = bosc_ame.spm.view {bank = 0 : i32, offset = 8192 : i64} : memref<16384xi8, #bosc_ame.spm<0>>
    %acc = memref.view %acc_buf[%c0][] : memref<16384xi8, #bosc_ame.spm<0>> to memref<64x64xi32, #bosc_ame.spm<0>>
    %tc = bosc_ame.tile.load_acc %acc : memref<64x64xi32, #bosc_ame.spm<0>> -> vector<64x64xi32>
    %r = bosc_ame.tile.mma %tc, %ta, %ta : vector<64x64xi32>, vector<64x64xi8>, vector<64x64xi8> -> vector<64x64xi32>
    bosc_ame.tile.store_acc %r, %acc : vector<64x64xi32>, memref<64x64xi32, #bosc_ame.spm<0>>
    return %c0_i32 : i32
  }
}
