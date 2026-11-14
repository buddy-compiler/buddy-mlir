// RUN: not buddy-opt %s --lower-bosc-ame 2>&1 | FileCheck %s --check-prefix=ERR
// RUN: not buddy-opt %s --lower-bosc-ame-tile 2>&1 | FileCheck %s --check-prefix=ERR

// The tile operations lower to the FPGA intrinsic set, but the module does not
// select an AME hardware contract.  Feeding the wrong contract to the hardware
// is silent (both encodings are valid `mtype` values), so the lowering demands
// an explicit choice instead of assuming one.

// ERR: does not select an AME hardware contract
// ERR-SAME: bosc_ame.target = "nr-fpga"

func.func @missing_target(%a: memref<4096xi8>) -> i32 {
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
