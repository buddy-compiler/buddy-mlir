// RUN: buddy-opt -verify-diagnostics %s --lower-bosc-ame

// The tile operations declare the memory effects the SPM/DMA checker needs:
// `tile.load_*` reads its buffer and `tile.store_acc` writes its buffer.  While
// a DMA request is in flight its destination is unavailable (not even for a
// read) and its source is read-only, so a tile access that overlaps the
// transfer is rejected before the SPM annotation is lost.  After the matching
// `bosc_ame.dma.wait` the same access is legal - which is what the first
// function below shows.  The check stops at the first failing function, so the
// write-side variant lives in `lower-tile-dma-store-order.mlir`.

module attributes {bosc_ame.target = "nr-fpga",
                   bosc_ame.dma.abi = "args_aligned64_nonalias"} {
  // Legal: stage A, wait, then load.  The checker runs functions in order, so
  // this one has to be verified first.
  func.func @wait_then_load(%src: memref<4096xi8>) -> i32 {
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %c4096 = arith.constant 4096 : index
    %buf = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64} : memref<16384xi8, #bosc_ame.spm<0>>
    %a = memref.view %buf[%c0][] : memref<16384xi8, #bosc_ame.spm<0>> to memref<64x64xi8, #bosc_ame.spm<0>>
    %acc = memref.view %buf[%c0][] : memref<16384xi8, #bosc_ame.spm<0>> to memref<64x64xi32, #bosc_ame.spm<0>>
    %t = bosc_ame.dma.start %src, %buf, %c4096 {channel = 0 : i32} : (memref<4096xi8>, memref<16384xi8, #bosc_ame.spm<0>>, index) -> !bosc_ame.dma_token<0>
    bosc_ame.dma.wait %t : !bosc_ame.dma_token<0>
    %tc = bosc_ame.tile.load_acc %acc : memref<64x64xi32, #bosc_ame.spm<0>> -> vector<64x64xi32>
    %ta = bosc_ame.tile.load_a %a : memref<64x64xi8, #bosc_ame.spm<0>> -> vector<64x64xi8>
    %r = bosc_ame.tile.mma %tc, %ta, %ta : vector<64x64xi32>, vector<64x64xi8>, vector<64x64xi8> -> vector<64x64xi32>
    bosc_ame.tile.store_acc %r, %acc : vector<64x64xi32>, memref<64x64xi32, #bosc_ame.spm<0>>
    return %c0_i32 : i32
  }

  // Illegal: the transfer into the A window has not been waited yet, so the
  // destination cannot be read.
  func.func @load_before_wait(%src: memref<4096xi8>) -> i32 {
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %c4096 = arith.constant 4096 : index
    %buf = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64} : memref<4096xi8, #bosc_ame.spm<0>>
    %a = memref.view %buf[%c0][] : memref<4096xi8, #bosc_ame.spm<0>> to memref<64x64xi8, #bosc_ame.spm<0>>
    %t = bosc_ame.dma.start %src, %buf, %c4096 {channel = 0 : i32} : (memref<4096xi8>, memref<4096xi8, #bosc_ame.spm<0>>, index) -> !bosc_ame.dma_token<0>
    // expected-error@+1 {{overlaps the destination of the in-flight request}}
    %ta = bosc_ame.tile.load_a %a : memref<64x64xi8, #bosc_ame.spm<0>> -> vector<64x64xi8>
    bosc_ame.dma.wait %t : !bosc_ame.dma_token<0>
    return %c0_i32 : i32
  }

}
