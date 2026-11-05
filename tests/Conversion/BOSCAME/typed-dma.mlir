// RUN: split-file %s %t
// RUN: buddy-opt %t/aligned.mlir --bosc-ame-spm-dma-check --lower-bosc-ame | FileCheck %s
// RUN: buddy-opt %t/aligned.mlir --lower-bosc-ame | buddy-translate --buddy-to-llvmir | FileCheck %s --check-prefix=LLVM
// RUN: not buddy-opt %t/overflow.mlir 2>&1 | FileCheck %s --check-prefix=BYTES
// RUN: not buddy-opt %t/unknown.mlir 2>&1 | FileCheck %s --check-prefix=ALIGN
// RUN: not buddy-opt %t/misaligned.mlir 2>&1 | FileCheck %s --check-prefix=ALIGN
// RUN: not buddy-opt %t/no-wait.mlir --bosc-ame-spm-dma-check 2>&1 | FileCheck %s --check-prefix=WAIT
// RUN: not buddy-opt %t/scalar.mlir --bosc-ame-spm-dma-check 2>&1 | FileCheck %s --check-prefix=SCALAR
//
// Typed DMA: legal aligned transfers lower to bosc_ame_dma_start; overflow,
// unproven alignment, missing waits, and scalar SPM access are rejected.

// CHECK: llvm.func @row
// CHECK: llvm.getelementptr {{.*}} : (!llvm.ptr, i64) -> !llvm.ptr, f32
// LLVM: getelementptr float
// LLVM: call i64 @bosc_ame_dma_start({{.*}}i64 3072)
// BYTES: transfer of 3136 bytes exceeds the src byte range of 3072 bytes
// ALIGN: endpoint effective address cannot be proven 64 B aligned
// WAIT: reads {{.*}} overlaps the destination of the in-flight request
// SCALAR: scalar load/store, ordinary memref.copy or Linalg may not access SPM

//--- aligned.mlir
module attributes {bosc_ame.dma.abi = "args_aligned64_nonalias"} {
  func.func @row(%src: memref<*xf32>, %row: index) {
    %c768 = arith.constant 768 : index
    %bytes = arith.constant 3072 : index
    %offset = arith.muli %row, %c768 : index
    %ddr = memref.reinterpret_cast %src to offset: [%offset], sizes: [768], strides: [1] : memref<*xf32> to memref<768xf32, strided<[1], offset: ?>>
    %spm = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64} : memref<768xf32, #bosc_ame.spm<0>>
    %t = bosc_ame.dma.start %ddr, %spm, %bytes {channel = 0 : i32} : (memref<768xf32, strided<[1], offset: ?>>, memref<768xf32, #bosc_ame.spm<0>>, index) -> !bosc_ame.dma_token<0>
    bosc_ame.dma.wait %t : !bosc_ame.dma_token<0>
    return
  }
}

//--- overflow.mlir
module attributes {bosc_ame.dma.abi = "args_aligned64_nonalias"} {
  func.func @overflow(%src: memref<768xf32>) {
    %bytes = arith.constant 3136 : index
    %spm = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64} : memref<1024xf32, #bosc_ame.spm<0>>
    %t = bosc_ame.dma.start %src, %spm, %bytes {channel = 0 : i32} : (memref<768xf32>, memref<1024xf32, #bosc_ame.spm<0>>, index) -> !bosc_ame.dma_token<0>
    bosc_ame.dma.wait %t : !bosc_ame.dma_token<0>
    return
  }
}

//--- unknown.mlir
module attributes {bosc_ame.dma.abi = "args_aligned64_nonalias"} {
  func.func @unknown(%src: memref<*xf32>, %offset: index) {
    %bytes = arith.constant 64 : index
    %ddr = memref.reinterpret_cast %src to offset: [%offset], sizes: [16], strides: [1] : memref<*xf32> to memref<16xf32, strided<[1], offset: ?>>
    %spm = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64} : memref<16xf32, #bosc_ame.spm<0>>
    %t = bosc_ame.dma.start %ddr, %spm, %bytes {channel = 0 : i32} : (memref<16xf32, strided<[1], offset: ?>>, memref<16xf32, #bosc_ame.spm<0>>, index) -> !bosc_ame.dma_token<0>
    bosc_ame.dma.wait %t : !bosc_ame.dma_token<0>
    return
  }
}

//--- misaligned.mlir
module attributes {bosc_ame.dma.abi = "args_aligned64_nonalias"} {
  func.func @misaligned(%src: memref<*xf32>) {
    %bytes = arith.constant 64 : index
    %ddr = memref.reinterpret_cast %src to offset: [1], sizes: [16], strides: [1] : memref<*xf32> to memref<16xf32, strided<[1], offset: 1>>
    %spm = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64} : memref<16xf32, #bosc_ame.spm<0>>
    %t = bosc_ame.dma.start %ddr, %spm, %bytes {channel = 0 : i32} : (memref<16xf32, strided<[1], offset: 1>>, memref<16xf32, #bosc_ame.spm<0>>, index) -> !bosc_ame.dma_token<0>
    bosc_ame.dma.wait %t : !bosc_ame.dma_token<0>
    return
  }
}

//--- no-wait.mlir
module attributes {bosc_ame.dma.abi = "args_aligned64_nonalias"} {
  func.func @no_wait(%src: memref<16xf32>) {
    %bytes = arith.constant 64 : index
    %zero = arith.constant 0 : index
    %spm = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64} : memref<16xf32, #bosc_ame.spm<0>>
    %t = bosc_ame.dma.start %src, %spm, %bytes {channel = 0 : i32} : (memref<16xf32>, memref<16xf32, #bosc_ame.spm<0>>, index) -> !bosc_ame.dma_token<0>
    %v = vector.load %spm[%zero] : memref<16xf32, #bosc_ame.spm<0>>, vector<16xf32>
    bosc_ame.dma.wait %t : !bosc_ame.dma_token<0>
    return
  }
}

//--- scalar.mlir
module {
  func.func @scalar() -> f32 {
    %zero = arith.constant 0 : index
    %spm = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64} : memref<16xf32, #bosc_ame.spm<0>>
    %v = memref.load %spm[%zero] : memref<16xf32, #bosc_ame.spm<0>>
    return %v : f32
  }
}
