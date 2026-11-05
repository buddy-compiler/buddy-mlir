// RUN: buddy-opt %s --lower-bosc-ame | FileCheck %s --check-prefix=LLVM
// RUN: buddy-opt %s --lower-bosc-ame | buddy-translate --buddy-to-llvmir | FileCheck %s --check-prefix=IR

// A tile operand keeps the offsets of the whole view chain, and each of them is
// applied exactly once:
//
//   * `bosc_ame.spm.view {bank = 0, offset = 4096}` compiles the bank base plus
//     the view offset into the descriptor (0x20000000 + 4096 = 536875008),
//   * the `memref.subview` offset (8 rows of 64 bytes = 512 bytes) is part of
//     the descriptor of the subview and must be added *by the descriptor*, not
//     by the tile lowering - and not twice.
//
// So the pointer fed to the AME load is the descriptor's aligned pointer plus
// one `getelementptr` of 512 bytes, and the row stride is the pitched 64 bytes
// of the subview, not the tile width of 16.

module attributes {bosc_ame.target = "nr-fpga"} {
  func.func @offset_chain() -> i32 {
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32

    // Bank 0 at byte 4096: 8192 bytes = 128 rows of 64 bytes.
    %a_buf = bosc_ame.spm.view {bank = 0 : i32, offset = 4096 : i64} : memref<8192xi8, #bosc_ame.spm<0>>
    // The A tile is rows 8..23 of the pitched [128, 64] view: descriptor
    // offset 8 * 64 = 512 elements.
    %a_big = memref.reinterpret_cast %a_buf to offset: [0], sizes: [128, 64], strides: [64, 1] : memref<8192xi8, #bosc_ame.spm<0>> to memref<128x64xi8, strided<[64, 1], offset: 0>, #bosc_ame.spm<0>>
    %a_tile = memref.subview %a_big[8, 0] [16, 16] [1, 1] : memref<128x64xi8, strided<[64, 1], offset: 0>, #bosc_ame.spm<0>> to memref<16x16xi8, strided<[64, 1], offset: 512>, #bosc_ame.spm<0>>

    // B and C come from their own windows so that the A pointer is the only
    // interesting one here.
    %b_buf = bosc_ame.spm.view {bank = 1 : i32, offset = 0 : i64} : memref<256xi8, #bosc_ame.spm<1>>
    %b = memref.reinterpret_cast %b_buf to offset: [0], sizes: [16, 16], strides: [1, 16] : memref<256xi8, #bosc_ame.spm<1>> to memref<16x16xi8, strided<[1, 16], offset: 0>, #bosc_ame.spm<1>>
    %c_buf = bosc_ame.spm.view {bank = 0 : i32, offset = 16384 : i64} : memref<1024xi8, #bosc_ame.spm<0>>
    %c = memref.view %c_buf[%c0][] : memref<1024xi8, #bosc_ame.spm<0>> to memref<16x16xi32, #bosc_ame.spm<0>>

    // The descriptor's first field is the bank base plus the spm.view offset
    // (0x20000000 + 4096 = 536875008), the aligned pointer is the same plus
    // the descriptor offset as a `getelementptr` of 512, and the load takes
    // that pointer with a 64-byte row stride.
    // LLVM: llvm.mlir.constant(536875008 : i64) : i64
    // The offset is folded into a single pointer constant by the time the LLVM
    // IR is emitted: 536875008 + 512, applied once.
    // IR: call <vscale x 128 x i8> @llvm.riscv.bosc.ordered.load.a.nxv128i8(ptr inttoptr (i64 536875520 to ptr), i64 64, i64 16, i64 16, i64 65552)
    %ta = bosc_ame.tile.load_a %a_tile : memref<16x16xi8, strided<[64, 1], offset: 512>, #bosc_ame.spm<0>> -> vector<16x16xi8>
    %tb = bosc_ame.tile.load_b %b : memref<16x16xi8, strided<[1, 16], offset: 0>, #bosc_ame.spm<1>> -> vector<16x16xi8>
    %tc = bosc_ame.tile.load_acc %c : memref<16x16xi32, #bosc_ame.spm<0>> -> vector<16x16xi32>
    %acc = bosc_ame.tile.mma %tc, %ta, %tb : vector<16x16xi32>, vector<16x16xi8>, vector<16x16xi8> -> vector<16x16xi32>
    bosc_ame.tile.store_acc %acc, %c : vector<16x16xi32>, memref<16x16xi32, #bosc_ame.spm<0>>
    return %c0_i32 : i32
  }
}
