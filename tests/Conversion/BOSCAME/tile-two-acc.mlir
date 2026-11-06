// RUN: buddy-opt %s --lower-bosc-ame | buddy-translate --buddy-to-llvmir > %t.ll
// RUN: llc -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c,-v,+xboscame -verify-machineinstrs < %t.ll > %t.nov.s
// RUN: llc -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c,+v,+xboscame -verify-machineinstrs < %t.ll > %t.v.s
// RUN: %PYTHON %S/Inputs/check-ordered-tiles.py %t.nov.s
// RUN: %PYTHON %S/Inputs/check-ordered-tiles.py %t.v.s

// Two defining clears and loop-carried ACCs must remain distinct. A is reused
// across the two MMAs; loading B1 must not overwrite B0 before its MMA. All
// transfers use DDR directly, with no SPM allocation or initial C read.
module attributes {bosc_ame.target = "nr-fpga",
                   bosc_ame.dma.abi = "args_aligned64_nonalias"} {
  func.func @two_acc_loop(
      %a: memref<4x64xi8>,
      %b0: memref<64x64xi8, strided<[1, 64]>>,
      %b1: memref<64x64xi8, strided<[1, 64]>>,
      %c0: memref<4x64xi32>, %c1: memref<4x64xi32>) {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %two = arith.constant 2 : index
    %z0 = bosc_ame.tile.zero_acc : vector<4x64xi32>
    %z1 = bosc_ame.tile.zero_acc : vector<4x64xi32>
    %results:2 = scf.for %i = %zero to %two step %one
        iter_args(%acc0 = %z0, %acc1 = %z1)
        -> (vector<4x64xi32>, vector<4x64xi32>) {
      %av = bosc_ame.tile.load_a %a {bosc_ame.direct_ddr} : memref<4x64xi8> -> vector<4x64xi8>
      %bv0 = bosc_ame.tile.load_b %b0 {bosc_ame.direct_ddr} : memref<64x64xi8, strided<[1, 64]>> -> vector<64x64xi8>
      %r0 = bosc_ame.tile.mma %acc0, %av, %bv0 : vector<4x64xi32>, vector<4x64xi8>, vector<64x64xi8> -> vector<4x64xi32>
      %bv1 = bosc_ame.tile.load_b %b1 {bosc_ame.direct_ddr} : memref<64x64xi8, strided<[1, 64]>> -> vector<64x64xi8>
      %r1 = bosc_ame.tile.mma %acc1, %av, %bv1 : vector<4x64xi32>, vector<4x64xi8>, vector<64x64xi8> -> vector<4x64xi32>
      scf.yield %r0, %r1 : vector<4x64xi32>, vector<4x64xi32>
    }
    bosc_ame.tile.store_acc %results#0, %c0 {bosc_ame.direct_ddr} : vector<4x64xi32>, memref<4x64xi32>
    bosc_ame.tile.store_acc %results#1, %c1 {bosc_ame.direct_ddr} : vector<4x64xi32>, memref<4x64xi32>
    return
  }
}
