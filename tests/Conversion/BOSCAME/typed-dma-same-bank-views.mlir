// RUN: split-file %s %t
// RUN: buddy-opt %t/valid.mlir --bosc-ame-spm-dma-check -o /dev/null
// RUN: buddy-opt %t/valid.mlir --lower-bosc-ame -o /dev/null
// RUN: buddy-opt %t/full-bank-argument.mlir --bosc-ame-spm-dma-check -o /dev/null
// RUN: buddy-opt %t/metadata.mlir --bosc-ame-spm-dma-check --lower-bosc-ame -o /dev/null
// RUN: not buddy-opt %t/metadata-overlap.mlir 2>&1 | FileCheck %s --check-prefix=METADATA-OVERLAP
// RUN: not buddy-opt %t/overlap.mlir 2>&1 | FileCheck %s --check-prefix=OVERLAP
// RUN: not buddy-opt %t/outside-root.mlir 2>&1 | FileCheck %s --check-prefix=BOUNDS
// RUN: not buddy-opt %t/negative-shift.mlir 2>&1 | FileCheck %s --check-prefix=BOUNDS
// RUN: not buddy-opt %t/overflow-shift.mlir 2>&1 | FileCheck %s --check-prefix=BOUNDS
// RUN: not buddy-opt %t/dynamic-shift.mlir 2>&1 | FileCheck %s --check-prefix=BOUNDS
// RUN: not buddy-opt %t/narrow-argument.mlir 2>&1 | FileCheck %s --check-prefix=BOUNDS
//
// Same-bank SPM views may DMA when ranges are disjoint and statically bounded
// inside a root view or full-bank argument; overlaps and unbound shifts fail.

// OVERLAP: src range [256, 320) and dst range [256, 320) overlap in SPM bank 0
// METADATA-OVERLAP: src range [512, 576) and dst range [512, 576) overlap in SPM bank 0
// BOUNDS: cannot statically bound an endpoint within its bosc_ame.spm.view or full-bank SPM block argument

//--- valid.mlir
module {
  func.func @replace_descriptor_offset() {
    %bytes = arith.constant 64 : index
    %root = bosc_ame.spm.view {bank = 0 : i32, offset = 64 : i64}
        : memref<256xi8, #bosc_ame.spm<0>>
    %parent = memref.subview %root[64] [128] [1]
        : memref<256xi8, #bosc_ame.spm<0>>
          to memref<128xi8, strided<[1], offset: 64>, #bosc_ame.spm<0>>
    // Reinterpret replaces the parent's descriptor offset; it is not added.
    %src = memref.reinterpret_cast %parent to offset: [0], sizes: [64], strides: [1]
        : memref<128xi8, strided<[1], offset: 64>, #bosc_ame.spm<0>>
          to memref<64xi8, #bosc_ame.spm<0>>
    %dst = memref.subview %root[128] [64] [1]
        : memref<256xi8, #bosc_ame.spm<0>>
          to memref<64xi8, strided<[1], offset: 128>, #bosc_ame.spm<0>>
    %t = bosc_ame.dma.start %src, %dst, %bytes {channel = 0 : i32}
        : (memref<64xi8, #bosc_ame.spm<0>>,
           memref<64xi8, strided<[1], offset: 128>, #bosc_ame.spm<0>>, index)
          -> !bosc_ame.dma_token<0>
    bosc_ame.dma.wait %t : !bosc_ame.dma_token<0>
    return
  }
  func.func @typed_pointer_shift() {
    %c64 = arith.constant 64 : index
    %root = bosc_ame.spm.view {bank = 0 : i32, offset = 128 : i64}
        : memref<256xi8, #bosc_ame.spm<0>>
    %typed = memref.view %root[%c64][]
        : memref<256xi8, #bosc_ame.spm<0>>
          to memref<32xf32, #bosc_ame.spm<0>>
    // The 64-byte pointer shift remains while the f32 offset contributes 64 B.
    %src = memref.reinterpret_cast %typed to offset: [16], sizes: [16], strides: [1]
        : memref<32xf32, #bosc_ame.spm<0>>
          to memref<16xf32, strided<[1], offset: 16>, #bosc_ame.spm<0>>
    %dst = memref.subview %root[192] [64] [1]
        : memref<256xi8, #bosc_ame.spm<0>>
          to memref<64xi8, strided<[1], offset: 192>, #bosc_ame.spm<0>>
    %t = bosc_ame.dma.start %src, %dst, %c64 {channel = 1 : i32}
        : (memref<16xf32, strided<[1], offset: 16>, #bosc_ame.spm<0>>,
           memref<64xi8, strided<[1], offset: 192>, #bosc_ame.spm<0>>, index)
          -> !bosc_ame.dma_token<1>
    bosc_ame.dma.wait %t : !bosc_ame.dma_token<1>
    return
  }
  func.func @shape_and_space_casts() {
    %bytes = arith.constant 64 : index
    %root = bosc_ame.spm.view {bank = 0 : i32, offset = 256 : i64}
        : memref<256xi8, #bosc_ame.spm<0>>
    %matrix = memref.expand_shape %root [[0, 1]] output_shape [4, 64]
        : memref<256xi8, #bosc_ame.spm<0>> into memref<4x64xi8, #bosc_ame.spm<0>>
    %flat = memref.collapse_shape %matrix [[0, 1]]
        : memref<4x64xi8, #bosc_ame.spm<0>> into memref<256xi8, #bosc_ame.spm<0>>
    %ordinary = memref.memory_space_cast %flat
        : memref<256xi8, #bosc_ame.spm<0>> to memref<256xi8>
    %spm = memref.memory_space_cast %ordinary
        : memref<256xi8> to memref<256xi8, #bosc_ame.spm<0>>
    %cast = memref.cast %spm : memref<256xi8, #bosc_ame.spm<0>>
        to memref<256xi8, strided<[1], offset: ?>, #bosc_ame.spm<0>>
    %src = memref.reinterpret_cast %cast to offset: [64], sizes: [64], strides: [1]
        : memref<256xi8, strided<[1], offset: ?>, #bosc_ame.spm<0>>
          to memref<64xi8, strided<[1], offset: 64>, #bosc_ame.spm<0>>
    %dst = memref.subview %root[192] [64] [1]
        : memref<256xi8, #bosc_ame.spm<0>>
          to memref<64xi8, strided<[1], offset: 192>, #bosc_ame.spm<0>>
    %t = bosc_ame.dma.start %src, %dst, %bytes {channel = 0 : i32}
        : (memref<64xi8, strided<[1], offset: 64>, #bosc_ame.spm<0>>,
           memref<64xi8, strided<[1], offset: 192>, #bosc_ame.spm<0>>, index)
          -> !bosc_ame.dma_token<0>
    bosc_ame.dma.wait %t : !bosc_ame.dma_token<0>
    return
  }
}

//--- full-bank-argument.mlir
// The checker retains the historical bank-root proof. LLVM export separately
// rejects SPM function parameters under the public ABI contract.
module {
  func.func @full_bank_argument(%root: memref<131072xi8, #bosc_ame.spm<0>>) {
    %bytes = arith.constant 64 : index
    %src = memref.subview %root[64] [64] [1]
        : memref<131072xi8, #bosc_ame.spm<0>>
          to memref<64xi8, strided<[1], offset: 64>, #bosc_ame.spm<0>>
    %dst = memref.reinterpret_cast %root to offset: [128], sizes: [64], strides: [1]
        : memref<131072xi8, #bosc_ame.spm<0>>
          to memref<64xi8, strided<[1], offset: 128>, #bosc_ame.spm<0>>
    %t = bosc_ame.dma.start %src, %dst, %bytes {channel = 0 : i32}
        : (memref<64xi8, strided<[1], offset: 64>, #bosc_ame.spm<0>>,
           memref<64xi8, strided<[1], offset: 128>, #bosc_ame.spm<0>>, index)
          -> !bosc_ame.dma_token<0>
    bosc_ame.dma.wait %t : !bosc_ame.dma_token<0>
    return
  }
}

//--- overlap.mlir
module {
  func.func @overlap() {
    %c64 = arith.constant 64 : index
    %root = bosc_ame.spm.view {bank = 0 : i32, offset = 128 : i64}
        : memref<256xi8, #bosc_ame.spm<0>>
    %typed = memref.view %root[%c64][]
        : memref<256xi8, #bosc_ame.spm<0>>
          to memref<32xf32, #bosc_ame.spm<0>>
    %src = memref.reinterpret_cast %typed to offset: [16], sizes: [16], strides: [1]
        : memref<32xf32, #bosc_ame.spm<0>>
          to memref<16xf32, strided<[1], offset: 16>, #bosc_ame.spm<0>>
    %dst = memref.subview %root[128] [64] [1]
        : memref<256xi8, #bosc_ame.spm<0>>
          to memref<64xi8, strided<[1], offset: 128>, #bosc_ame.spm<0>>
    %t = bosc_ame.dma.start %src, %dst, %c64 {channel = 0 : i32}
        : (memref<16xf32, strided<[1], offset: 16>, #bosc_ame.spm<0>>,
           memref<64xi8, strided<[1], offset: 128>, #bosc_ame.spm<0>>, index)
          -> !bosc_ame.dma_token<0>
    bosc_ame.dma.wait %t : !bosc_ame.dma_token<0>
    return
  }
}

//--- outside-root.mlir
module {
  func.func @outside_root() {
    %bytes = arith.constant 64 : index
    %root = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64}
        : memref<64xi8, #bosc_ame.spm<0>>
    %src = memref.reinterpret_cast %root to offset: [64], sizes: [64], strides: [1]
        : memref<64xi8, #bosc_ame.spm<0>>
          to memref<64xi8, strided<[1], offset: 64>, #bosc_ame.spm<0>>
    %dst = bosc_ame.spm.view {bank = 0 : i32, offset = 256 : i64}
        : memref<64xi8, #bosc_ame.spm<0>>
    %t = bosc_ame.dma.start %src, %dst, %bytes {channel = 0 : i32}
        : (memref<64xi8, strided<[1], offset: 64>, #bosc_ame.spm<0>>,
           memref<64xi8, #bosc_ame.spm<0>>, index) -> !bosc_ame.dma_token<0>
    bosc_ame.dma.wait %t : !bosc_ame.dma_token<0>
    return
  }
}

//--- negative-shift.mlir
module {
  func.func @negative_shift() {
    %bytes = arith.constant 64 : index
    %negative = arith.constant -64 : index
    %root = bosc_ame.spm.view {bank = 0 : i32, offset = 64 : i64}
        : memref<64xi8, #bosc_ame.spm<0>>
    %src = memref.view %root[%negative][]
        : memref<64xi8, #bosc_ame.spm<0>> to memref<64xi8, #bosc_ame.spm<0>>
    %dst = bosc_ame.spm.view {bank = 0 : i32, offset = 256 : i64}
        : memref<64xi8, #bosc_ame.spm<0>>
    %t = bosc_ame.dma.start %src, %dst, %bytes {channel = 0 : i32}
        : (memref<64xi8, #bosc_ame.spm<0>>, memref<64xi8, #bosc_ame.spm<0>>, index)
          -> !bosc_ame.dma_token<0>
    bosc_ame.dma.wait %t : !bosc_ame.dma_token<0>
    return
  }
}

//--- overflow-shift.mlir
module {
  func.func @overflow_shift() {
    %bytes = arith.constant 64 : index
    %huge = arith.constant 9223372036854775744 : index
    %root = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64}
        : memref<64xi8, #bosc_ame.spm<0>>
    %view = memref.view %root[%huge][]
        : memref<64xi8, #bosc_ame.spm<0>> to memref<64xi8, #bosc_ame.spm<0>>
    %src = memref.view %view[%bytes][]
        : memref<64xi8, #bosc_ame.spm<0>> to memref<64xi8, #bosc_ame.spm<0>>
    %dst = bosc_ame.spm.view {bank = 0 : i32, offset = 256 : i64}
        : memref<64xi8, #bosc_ame.spm<0>>
    %t = bosc_ame.dma.start %src, %dst, %bytes {channel = 0 : i32}
        : (memref<64xi8, #bosc_ame.spm<0>>, memref<64xi8, #bosc_ame.spm<0>>, index)
          -> !bosc_ame.dma_token<0>
    bosc_ame.dma.wait %t : !bosc_ame.dma_token<0>
    return
  }
}

//--- dynamic-shift.mlir
module {
  func.func @dynamic_shift(%row: index) {
    %bytes = arith.constant 64 : index
    %shift = arith.muli %row, %bytes : index
    %root = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64}
        : memref<256xi8, #bosc_ame.spm<0>>
    %src = memref.view %root[%shift][]
        : memref<256xi8, #bosc_ame.spm<0>> to memref<64xi8, #bosc_ame.spm<0>>
    %dst = bosc_ame.spm.view {bank = 0 : i32, offset = 512 : i64}
        : memref<64xi8, #bosc_ame.spm<0>>
    %t = bosc_ame.dma.start %src, %dst, %bytes {channel = 0 : i32}
        : (memref<64xi8, #bosc_ame.spm<0>>, memref<64xi8, #bosc_ame.spm<0>>, index)
          -> !bosc_ame.dma_token<0>
    bosc_ame.dma.wait %t : !bosc_ame.dma_token<0>
    return
  }
}

//--- narrow-argument.mlir
module {
  func.func @narrow_argument(%root: memref<256xi8, #bosc_ame.spm<0>>) {
    %bytes = arith.constant 64 : index
    %src = memref.reinterpret_cast %root to offset: [0], sizes: [64], strides: [1]
        : memref<256xi8, #bosc_ame.spm<0>> to memref<64xi8, #bosc_ame.spm<0>>
    %dst = bosc_ame.spm.view {bank = 0 : i32, offset = 512 : i64}
        : memref<64xi8, #bosc_ame.spm<0>>
    %t = bosc_ame.dma.start %src, %dst, %bytes {channel = 0 : i32}
        : (memref<64xi8, #bosc_ame.spm<0>>, memref<64xi8, #bosc_ame.spm<0>>, index)
          -> !bosc_ame.dma_token<0>
    bosc_ame.dma.wait %t : !bosc_ame.dma_token<0>
    return
  }
}

//--- metadata.mlir
module {
  func.func @metadata_pointer_origin() {
    %bytes = arith.constant 64 : index
    %root = bosc_ame.spm.view {bank = 0 : i32, offset = 512 : i64}
        : memref<256xi8, #bosc_ame.spm<0>>
    %parent = memref.subview %root[64] [128] [1]
        : memref<256xi8, #bosc_ame.spm<0>>
          to memref<128xi8, strided<[1], offset: 64>, #bosc_ame.spm<0>>
    %base, %offset, %size, %stride = memref.extract_strided_metadata %parent
        : memref<128xi8, strided<[1], offset: 64>, #bosc_ame.spm<0>>
          -> memref<i8, #bosc_ame.spm<0>>, index, index, index
    // The base buffer retains the aligned pointer, without the parent's offset.
    %src = memref.reinterpret_cast %base to offset: [0], sizes: [64], strides: [1]
        : memref<i8, #bosc_ame.spm<0>> to memref<64xi8, #bosc_ame.spm<0>>
    %dst = memref.subview %root[128] [64] [1]
        : memref<256xi8, #bosc_ame.spm<0>> to memref<64xi8, strided<[1], offset: 128>, #bosc_ame.spm<0>>
    %t = bosc_ame.dma.start %src, %dst, %bytes {channel = 0 : i32}
        : (memref<64xi8, #bosc_ame.spm<0>>, memref<64xi8, strided<[1], offset: 128>, #bosc_ame.spm<0>>, index)
          -> !bosc_ame.dma_token<0>
    bosc_ame.dma.wait %t : !bosc_ame.dma_token<0>
    return
  }
}

//--- metadata-overlap.mlir
module {
  func.func @metadata_pointer_origin() {
    %bytes = arith.constant 64 : index
    %root = bosc_ame.spm.view {bank = 0 : i32, offset = 512 : i64}
        : memref<256xi8, #bosc_ame.spm<0>>
    %parent = memref.subview %root[64] [128] [1]
        : memref<256xi8, #bosc_ame.spm<0>>
          to memref<128xi8, strided<[1], offset: 64>, #bosc_ame.spm<0>>
    %base, %offset, %size, %stride = memref.extract_strided_metadata %parent
        : memref<128xi8, strided<[1], offset: 64>, #bosc_ame.spm<0>>
          -> memref<i8, #bosc_ame.spm<0>>, index, index, index
    // The base buffer retains the aligned pointer, without the parent's offset.
    %src = memref.reinterpret_cast %base to offset: [0], sizes: [64], strides: [1]
        : memref<i8, #bosc_ame.spm<0>> to memref<64xi8, #bosc_ame.spm<0>>
    %dst = memref.subview %root[0] [64] [1]
        : memref<256xi8, #bosc_ame.spm<0>> to memref<64xi8, #bosc_ame.spm<0>>
    %t = bosc_ame.dma.start %src, %dst, %bytes {channel = 0 : i32}
        : (memref<64xi8, #bosc_ame.spm<0>>, memref<64xi8, #bosc_ame.spm<0>>, index)
          -> !bosc_ame.dma_token<0>
    bosc_ame.dma.wait %t : !bosc_ame.dma_token<0>
    return
  }
}
