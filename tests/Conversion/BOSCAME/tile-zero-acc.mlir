// RUN: buddy-opt %s --lower-bosc-ame | FileCheck %s --check-prefix=LOWER
// RUN: buddy-opt %s --lower-bosc-ame | buddy-translate --buddy-to-llvmir | llc -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c,-v,+xboscame -filetype=asm -o - | FileCheck %s --check-prefix=ASM
//
// `tile.zero_acc` lowers to the ordered zero intrinsic (self-clear mwsub), not
// an accumulator memory load; the LLC path must emit msettile*/msettype and
// mwsub.h.mm without mlce32.m.

// LOWER: llvm.call_intrinsic "llvm.riscv.bosc.ordered.zero.acc"
// LOWER-NOT: llvm.inline_asm
// LOWER-NOT: ordered.load.acc
// ASM-LABEL: {{^}}zero_product:
// ASM-NOT: mlce32.m
// ASM: fence rw, rw
// ASM: msettilem zero,
// ASM-NEXT: msettilen zero,
// ASM-NEXT: msettype zero,
// ASM-NEXT: mwsub.h.mm [[ACC:acc[0-7]]], [[ACC]], [[ACC]]
// ASM-NEXT: fence rw, rw
// ASM: mlae8.m [[A:tr[0-7]]],
// ASM: mlbe8.m [[B:tr[0-7]]],
// ASM: mqma.b.mm [[ACC]], [[A]], [[B]]
// ASM: msce32.m [[ACC]],
// ASM-NOT: mlce32.m
module attributes {bosc_ame.target = "nr-fpga"} {
  func.func @zero_product() {
    %a = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64} : memref<32x64xi8, #bosc_ame.spm<0>>
    %rawb = bosc_ame.spm.view {bank = 1 : i32, offset = 0 : i64} : memref<4096xi8, #bosc_ame.spm<1>>
    %b = memref.reinterpret_cast %rawb to offset: [0], sizes: [64, 64], strides: [1, 64] : memref<4096xi8, #bosc_ame.spm<1>> to memref<64x64xi8, strided<[1, 64]>, #bosc_ame.spm<1>>
    %c = bosc_ame.spm.view {bank = 0 : i32, offset = 4096 : i64} : memref<32x64xi32, #bosc_ame.spm<0>>
    %z = bosc_ame.tile.zero_acc : vector<32x64xi32>
    %ta = bosc_ame.tile.load_a %a : memref<32x64xi8, #bosc_ame.spm<0>> -> vector<32x64xi8>
    %tb = bosc_ame.tile.load_b %b : memref<64x64xi8, strided<[1, 64]>, #bosc_ame.spm<1>> -> vector<64x64xi8>
    %acc = bosc_ame.tile.mma %z, %ta, %tb {bosc_ame.ordered} : vector<32x64xi32>, vector<32x64xi8>, vector<64x64xi8> -> vector<32x64xi32>
    bosc_ame.tile.store_acc %acc, %c : vector<32x64xi32>, memref<32x64xi32, #bosc_ame.spm<0>>
    return
  }
}
