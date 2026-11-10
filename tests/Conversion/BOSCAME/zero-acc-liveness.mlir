// RUN: buddy-opt %s --split-input-file --verify-diagnostics --bosc-ame-spm-dma-check
//
// Live `tile.zero_acc` values share the AME accumulator liveness rules with
// ordinary ACC defs: no DMA start, no nested zero, and no calls while live.

module attributes {bosc_ame.target = "nr-fpga", bosc_ame.dma.abi = "args_aligned64_nonalias"} {
  func.func @start_while_zero_live(%src: memref<64xi8>) {
    %bytes = arith.constant 64 : index
    %dma = bosc_ame.spm.view {bank = 1 : i32, offset = 0 : i64} : memref<64xi8, #bosc_ame.spm<1>>
    %out = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64} : memref<1x1xi32, #bosc_ame.spm<0>>
    %z = bosc_ame.tile.zero_acc : vector<1x1xi32>
    // expected-error @+1 {{cannot start a DMA request while an AME accumulator is live}}
    %t = bosc_ame.dma.start %src, %dma, %bytes {channel = 0 : i32} : (memref<64xi8>, memref<64xi8, #bosc_ame.spm<1>>, index) -> !bosc_ame.dma_token<0>
    bosc_ame.dma.wait %t : !bosc_ame.dma_token<0>
    bosc_ame.tile.store_acc %z, %out : vector<1x1xi32>, memref<1x1xi32, #bosc_ame.spm<0>>
    return
  }
}

// -----

module attributes {bosc_ame.target = "nr-fpga", bosc_ame.dma.abi = "args_aligned64_nonalias"} {
  func.func @zero_before_wait(%src: memref<64xi8>) {
    %bytes = arith.constant 64 : index
    %dma = bosc_ame.spm.view {bank = 1 : i32, offset = 0 : i64} : memref<64xi8, #bosc_ame.spm<1>>
    %out = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64} : memref<1x1xi32, #bosc_ame.spm<0>>
    %t = bosc_ame.dma.start %src, %dma, %bytes {channel = 0 : i32} : (memref<64xi8>, memref<64xi8, #bosc_ame.spm<1>>, index) -> !bosc_ame.dma_token<0>
    // expected-error @+1 {{cannot start a zero accumulator while a DMA request is in flight}}
    %z = bosc_ame.tile.zero_acc : vector<1x1xi32>
    bosc_ame.dma.wait %t : !bosc_ame.dma_token<0>
    bosc_ame.tile.store_acc %z, %out : vector<1x1xi32>, memref<1x1xi32, #bosc_ame.spm<0>>
    return
  }
}

// -----

module attributes {bosc_ame.target = "nr-fpga"} {
  func.func private @callee()
  func.func @call_while_zero_live() {
    %out = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64} : memref<1x1xi32, #bosc_ame.spm<0>>
    %z = bosc_ame.tile.zero_acc : vector<1x1xi32>
    // expected-error @+1 {{cannot call code while an AME accumulator is live}}
    func.call @callee() : () -> ()
    bosc_ame.tile.store_acc %z, %out : vector<1x1xi32>, memref<1x1xi32, #bosc_ame.spm<0>>
    return
  }
}

// -----

module attributes {bosc_ame.target = "nr-fpga", bosc_ame.dma.abi = "args_aligned64_nonalias"} {
  func.func private @callee()
  func.func @complete_before_dma_and_call(%src: memref<64xi8>) {
    %bytes = arith.constant 64 : index
    %dma = bosc_ame.spm.view {bank = 1 : i32, offset = 0 : i64} : memref<64xi8, #bosc_ame.spm<1>>
    %out = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64} : memref<1x1xi32, #bosc_ame.spm<0>>
    %z = bosc_ame.tile.zero_acc : vector<1x1xi32>
    bosc_ame.tile.store_acc %z, %out : vector<1x1xi32>, memref<1x1xi32, #bosc_ame.spm<0>>
    %t = bosc_ame.dma.start %src, %dma, %bytes {channel = 0 : i32} : (memref<64xi8>, memref<64xi8, #bosc_ame.spm<1>>, index) -> !bosc_ame.dma_token<0>
    bosc_ame.dma.wait %t : !bosc_ame.dma_token<0>
    func.call @callee() : () -> ()
    return
  }
}
