// RUN: split-file %s %t
// RUN: buddy-opt %t/serial.mlir --bosc-ame-spm-dma-check --lower-bosc-ame -o /dev/null
// RUN: buddy-opt %t/alias.mlir --bosc-ame-spm-dma-check -verify-diagnostics -o /dev/null

// Alignment and DDR disjointness are separate facts. A synchronous DDR/SPM
// transfer needs alignment alone; outstanding accesses still require the
// ordinary alias proof and cannot assume distinct arguments do not overlap.

//--- serial.mlir
module attributes {bosc_ame.dma.abi = "args_aligned64"} {
  func.func @serial(%src: memref<64xi8>, %dst: memref<64xi8>) {
    %bytes = arith.constant 64 : index
    %spm = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64}
        : memref<64xi8, #bosc_ame.spm<0>>
    %load = bosc_ame.dma.start %src, %spm, %bytes {channel = 0 : i32}
        : (memref<64xi8>, memref<64xi8, #bosc_ame.spm<0>>, index)
          -> !bosc_ame.dma_token<0>
    bosc_ame.dma.wait %load : !bosc_ame.dma_token<0>
    %store = bosc_ame.dma.start %spm, %dst, %bytes {channel = 0 : i32}
        : (memref<64xi8, #bosc_ame.spm<0>>, memref<64xi8>, index)
          -> !bosc_ame.dma_token<0>
    bosc_ame.dma.wait %store : !bosc_ame.dma_token<0>
    return
  }
}

//--- alias.mlir
module attributes {bosc_ame.dma.abi = "args_aligned64"} {
  func.func @alias(%src: memref<64xi8>, %dst: memref<64xi8>) {
    %bytes = arith.constant 64 : index
    %spm0 = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64}
        : memref<64xi8, #bosc_ame.spm<0>>
    %spm1 = bosc_ame.spm.view {bank = 1 : i32, offset = 0 : i64}
        : memref<64xi8, #bosc_ame.spm<1>>
    %load = bosc_ame.dma.start %src, %spm0, %bytes {channel = 0 : i32}
        : (memref<64xi8>, memref<64xi8, #bosc_ame.spm<0>>, index)
          -> !bosc_ame.dma_token<0>
    // expected-error@+1 {{overlaps the in-flight request on DMA channel}}
    %store = bosc_ame.dma.start %spm1, %dst, %bytes {channel = 1 : i32}
        : (memref<64xi8, #bosc_ame.spm<1>>, memref<64xi8>, index)
          -> !bosc_ame.dma_token<1>
    bosc_ame.dma.wait %load : !bosc_ame.dma_token<0>
    bosc_ame.dma.wait %store : !bosc_ame.dma_token<1>
    return
  }
}
