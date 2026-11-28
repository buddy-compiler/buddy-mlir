// RUN: buddy-opt %s -lower-xsai-ame | FileCheck %s

// Complete tiles configure every dimension with immediate constants, in
// logical tile order, even when the memref view itself is dynamic. Partial,
// dynamic, and size-overridden tiles are rejected (see unsupported.mlir).
// CHECK-LABEL: func.func @dynamic_view_a
// CHECK: memref.extract_strided_metadata
// CHECK: %[[M:.*]] = llvm.mlir.constant(128 : i64)
// CHECK: llvm.call_intrinsic "llvm.riscv.xsai.zames.msettilemi"(%[[M]]) : (i64) -> ()
// CHECK: %[[K:.*]] = llvm.mlir.constant(64 : i64)
// CHECK: llvm.call_intrinsic "llvm.riscv.xsai.zames.msettileki"(%[[K]]) : (i64) -> ()
// CHECK: llvm.call_intrinsic "llvm.riscv.xsai.zames.mla"
// CHECK-NOT: llvm.riscv.xsai.zames.msettile
// CHECK: llvm.call_intrinsic "llvm.riscv.xsai.zames.msa"
func.func @dynamic_view_a(%view: memref<?x?xi8>) {
  %a = xsai_ame.load %view : memref<?x?xi8> -> !xsai_ame.tile<a, 128x64, i8>
  xsai_ame.store %a, %view : !xsai_ame.tile<a, 128x64, i8>, memref<?x?xi8>
  return
}

// B configures N/K and C configures M/N; half elements use K = 32.
// CHECK-LABEL: func.func @role_axes
// CHECK: %[[N:.*]] = llvm.mlir.constant(128 : i64)
// CHECK: llvm.call_intrinsic "llvm.riscv.xsai.zames.msettileni"(%[[N]])
// CHECK: %[[K:.*]] = llvm.mlir.constant(32 : i64)
// CHECK: llvm.call_intrinsic "llvm.riscv.xsai.zames.msettileki"(%[[K]])
// CHECK: llvm.call_intrinsic "llvm.riscv.xsai.zames.mlb"
// CHECK: %[[CM:.*]] = llvm.mlir.constant(128 : i64)
// CHECK: llvm.call_intrinsic "llvm.riscv.xsai.zames.msettilemi"(%[[CM]])
// CHECK-NOT: llvm.riscv.xsai.zames.msettile
// CHECK: llvm.call_intrinsic "llvm.riscv.xsai.zames.mlc"
func.func @role_axes(%b: memref<128x32xf16>, %c: memref<128x128xf32>) {
  %bt = xsai_ame.load %b : memref<128x32xf16> -> !xsai_ame.tile<b, 128x32, fp16>
  %ct = xsai_ame.load %c : memref<128x128xf32> -> !xsai_ame.tile<c, 128x128, fp32>
  return
}

// Offset and major stride are runtime metadata, not assumed zero/contiguous.
// CHECK-LABEL: func.func @dynamic_metadata
// CHECK: memref.extract_strided_metadata
// CHECK: memref.extract_aligned_pointer_as_index
// CHECK: %[[WIDTH:.*]] = llvm.mlir.constant(2 : i64)
// CHECK: arith.index_cast
// CHECK: %[[OFFSET:.*]] = arith.muli {{.*}}, %[[WIDTH]]
// CHECK: %[[ADDRESS:.*]] = arith.addi {{.*}}, %[[OFFSET]]
// CHECK: llvm.inttoptr %[[ADDRESS]]
// CHECK: arith.index_cast
// CHECK: %[[STRIDE:.*]] = arith.muli {{.*}}, %[[WIDTH]]
// CHECK: llvm.call_intrinsic "llvm.riscv.xsai.zames.mlbt"({{.*}}, %[[STRIDE]])
func.func @dynamic_metadata(%view: memref<?x?xf16, strided<[?, 1], offset: ?>>) {
  %b = xsai_ame.load %view <{transpose = true}> : memref<?x?xf16, strided<[?, 1], offset: ?>> -> !xsai_ame.tile<b, 128x32, fp16>
  return
}

// A subview's offset must survive into extract_strided_metadata.
// CHECK-LABEL: func.func @subview_offset
// CHECK: %[[SUB:.*]] = memref.subview
// CHECK: memref.extract_strided_metadata %[[SUB]]
// CHECK: arith.muli
// CHECK: arith.addi
// CHECK: llvm.call_intrinsic "llvm.riscv.xsai.zames.mlc"
func.func @subview_offset(%base: memref<256x256xf32>) {
  %view = memref.subview %base[3, 5] [128, 128] [1, 1] : memref<256x256xf32> to memref<128x128xf32, strided<[256, 1], offset: 773>>
  %c = xsai_ame.load %view : memref<128x128xf32, strided<[256, 1], offset: 773>> -> !xsai_ame.tile<c, 128x128, fp32>
  xsai_ame.store %c, %view : !xsai_ame.tile<c, 128x128, fp32>, memref<128x128xf32, strided<[256, 1], offset: 773>>
  return
}

// Logical dynamic dimensions never configure a physical whole-register clear.
// CHECK-LABEL: func.func @whole_register_zero
// CHECK-NOT: llvm.riscv.xsai.zames.msettile
// CHECK: llvm.call_intrinsic "llvm.riscv.xsai.zames.mzero"
// CHECK-NOT: llvm.riscv.xsai.zames.msettile
// CHECK: llvm.call_intrinsic "llvm.riscv.xsai.zames.mzero"
// CHECK-NOT: llvm.riscv.xsai.zames.msettile
// CHECK: llvm.call_intrinsic "llvm.riscv.xsai.zames.mzero"
// CHECK-NOT: llvm.riscv.xsai.zames.msettile
// CHECK: return
func.func @whole_register_zero(%x: i64, %y: i64) {
  %a = xsai_ame.mzero[%x, %y] : !xsai_ame.tile<a, ?x?, i8>
  %b = xsai_ame.mzero[%x] : !xsai_ame.tile<b, ?x1024, bf16>
  %c = xsai_ame.mzero : !xsai_ame.tile<c, 1024x1, fp32>
  return
}
