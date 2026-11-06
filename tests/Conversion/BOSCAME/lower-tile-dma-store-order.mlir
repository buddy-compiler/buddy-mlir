// RUN: buddy-opt -verify-diagnostics %s --lower-bosc-ame

// The write side of the SPM/DMA contract: while a transfer is in flight its
// source may only be read, so a `bosc_ame.tile.store_acc` that writes the
// buffer a DMA engine is still reading is rejected.  The destination of such a
// transfer would be rejected as well, from the same `Write` effect.

module attributes {bosc_ame.target = "nr-fpga",
                   bosc_ame.dma.abi = "args_aligned64_nonalias"} {
  func.func @store_before_wait(%dst: memref<4096xi8>) -> i32 {
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %c4096 = arith.constant 4096 : index
    %buf = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64} : memref<16384xi8, #bosc_ame.spm<0>>
    // Keep the accumulator load disjoint from the DMA source.  The store
    // below still targets the source, so this test reaches the write-side
    // occupancy check after the new accumulator sequencing check permits the
    // load itself.
    %acc_buf = bosc_ame.spm.view {bank = 1 : i32, offset = 0 : i64} : memref<16384xi8, #bosc_ame.spm<1>>
    %acc = memref.view %acc_buf[%c0][] : memref<16384xi8, #bosc_ame.spm<1>> to memref<64x64xi32, #bosc_ame.spm<1>>
    %src_acc = memref.view %buf[%c0][] : memref<16384xi8, #bosc_ame.spm<0>> to memref<64x64xi32, #bosc_ame.spm<0>>
    %trim = memref.subview %src_acc[0, 0] [64, 64] [1, 1] : memref<64x64xi32, #bosc_ame.spm<0>> to memref<64x64xi32, strided<[64, 1], offset: 0>, #bosc_ame.spm<0>>
    %t = bosc_ame.dma.start %buf, %dst, %c4096 {channel = 0 : i32} : (memref<16384xi8, #bosc_ame.spm<0>>, memref<4096xi8>, index) -> !bosc_ame.dma_token<0>
    %value = bosc_ame.tile.load_acc %acc : memref<64x64xi32, #bosc_ame.spm<1>> -> vector<64x64xi32>
    // expected-error@+1 {{overlaps the source of the in-flight request}}
    bosc_ame.tile.store_acc %value, %trim : vector<64x64xi32>, memref<64x64xi32, strided<[64, 1], offset: 0>, #bosc_ame.spm<0>>
    bosc_ame.dma.wait %t : !bosc_ame.dma_token<0>
    return %c0_i32 : i32
  }
}
