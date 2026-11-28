// RUN: buddy-opt %s -lower-xsai-ame | FileCheck %s

// CHECK-LABEL: func.func @all_roles
// CHECK: llvm.call_intrinsic "llvm.riscv.xsai.zames.mla"{{.*}} -> !llvm.target<"riscv.xsai.zames.tile.int8">
// CHECK: llvm.call_intrinsic "llvm.riscv.xsai.zames.msa"
// CHECK: llvm.call_intrinsic "llvm.riscv.xsai.zames.mlat"
// CHECK: llvm.call_intrinsic "llvm.riscv.xsai.zames.msat"
// CHECK: llvm.call_intrinsic "llvm.riscv.xsai.zames.mlb"
// CHECK: llvm.call_intrinsic "llvm.riscv.xsai.zames.msb"
// CHECK: llvm.call_intrinsic "llvm.riscv.xsai.zames.mlbt"
// CHECK: llvm.call_intrinsic "llvm.riscv.xsai.zames.msbt"
// CHECK: llvm.call_intrinsic "llvm.riscv.xsai.zames.mlc"{{.*}} -> !llvm.target<"riscv.xsai.zames.tile.int32">
// CHECK: llvm.call_intrinsic "llvm.riscv.xsai.zames.msc"
// CHECK: llvm.call_intrinsic "llvm.riscv.xsai.zames.mlct"
// CHECK: llvm.call_intrinsic "llvm.riscv.xsai.zames.msct"
func.func @all_roles(%a: memref<128x64xi8>, %at: memref<64x128xi8>, %b: memref<128x64xi8>, %bt: memref<64x128xi8>, %c: memref<128x128xi32>, %ct: memref<128x128xi32>) {
  %a0 = xsai_ame.load %a : memref<128x64xi8> -> !xsai_ame.tile<a, 128x64, i8>
  xsai_ame.store %a0, %a : !xsai_ame.tile<a, 128x64, i8>, memref<128x64xi8>
  %a1 = xsai_ame.load %at <{transpose = true}> : memref<64x128xi8> -> !xsai_ame.tile<a, 128x64, i8>
  xsai_ame.store %a1, %at <{transpose = true}> : !xsai_ame.tile<a, 128x64, i8>, memref<64x128xi8>
  %b0 = xsai_ame.load %b : memref<128x64xi8> -> !xsai_ame.tile<b, 128x64, i8>
  xsai_ame.store %b0, %b : !xsai_ame.tile<b, 128x64, i8>, memref<128x64xi8>
  %b1 = xsai_ame.load %bt <{transpose = true}> : memref<64x128xi8> -> !xsai_ame.tile<b, 128x64, i8>
  xsai_ame.store %b1, %bt <{transpose = true}> : !xsai_ame.tile<b, 128x64, i8>, memref<64x128xi8>
  %c0 = xsai_ame.load %c : memref<128x128xi32> -> !xsai_ame.tile<c, 128x128, i32>
  xsai_ame.store %c0, %c : !xsai_ame.tile<c, 128x128, i32>, memref<128x128xi32>
  %c1 = xsai_ame.load %ct <{transpose = true}> : memref<128x128xi32> -> !xsai_ame.tile<c, 128x128, i32>
  xsai_ame.store %c1, %ct <{transpose = true}> : !xsai_ame.tile<c, 128x128, i32>, memref<128x128xi32>
  return
}

// 1/2/4-byte memory types have distinct byte row-stride scaling. Typed FP8
// memrefs remain FP8, rather than pretending FP8 payloads are integer bytes.
// CHECK-LABEL: func.func @typed_memory
// CHECK: llvm.mlir.constant(1 : i64)
// CHECK: llvm.call_intrinsic "llvm.riscv.xsai.zames.mla"{{.*}} -> !llvm.target<"riscv.xsai.zames.tile.fp8e4">
// CHECK: llvm.call_intrinsic "llvm.riscv.xsai.zames.msa"
// CHECK: llvm.call_intrinsic "llvm.riscv.xsai.zames.mlb"{{.*}} -> !llvm.target<"riscv.xsai.zames.tile.fp8e5">
// CHECK: llvm.call_intrinsic "llvm.riscv.xsai.zames.msb"
// CHECK: llvm.mlir.constant(2 : i64)
// CHECK: llvm.call_intrinsic "llvm.riscv.xsai.zames.mla"{{.*}} -> !llvm.target<"riscv.xsai.zames.tile.fp16">
// CHECK: llvm.call_intrinsic "llvm.riscv.xsai.zames.msa"
// CHECK: llvm.call_intrinsic "llvm.riscv.xsai.zames.mlb"{{.*}} -> !llvm.target<"riscv.xsai.zames.tile.bf16">
// CHECK: llvm.call_intrinsic "llvm.riscv.xsai.zames.msb"
// CHECK: llvm.mlir.constant(4 : i64)
// CHECK: llvm.call_intrinsic "llvm.riscv.xsai.zames.mlc"{{.*}} -> !llvm.target<"riscv.xsai.zames.tile.fp32">
// CHECK: llvm.call_intrinsic "llvm.riscv.xsai.zames.msc"
func.func @typed_memory(%e4: memref<128x64xf8E4M3FN>, %e5: memref<128x64xf8E5M2>, %half: memref<128x32xf16>, %brain: memref<128x32xbf16>, %single: memref<128x128xf32>) {
  %a4 = xsai_ame.load %e4 : memref<128x64xf8E4M3FN> -> !xsai_ame.tile<a, 128x64, fp8e4m3>
  xsai_ame.store %a4, %e4 : !xsai_ame.tile<a, 128x64, fp8e4m3>, memref<128x64xf8E4M3FN>
  %b5 = xsai_ame.load %e5 : memref<128x64xf8E5M2> -> !xsai_ame.tile<b, 128x64, fp8e5m2>
  xsai_ame.store %b5, %e5 : !xsai_ame.tile<b, 128x64, fp8e5m2>, memref<128x64xf8E5M2>
  %ah = xsai_ame.load %half : memref<128x32xf16> -> !xsai_ame.tile<a, 128x32, fp16>
  xsai_ame.store %ah, %half : !xsai_ame.tile<a, 128x32, fp16>, memref<128x32xf16>
  %bb = xsai_ame.load %brain : memref<128x32xbf16> -> !xsai_ame.tile<b, 128x32, bf16>
  xsai_ame.store %bb, %brain : !xsai_ame.tile<b, 128x32, bf16>, memref<128x32xbf16>
  %cs = xsai_ame.load %single : memref<128x128xf32> -> !xsai_ame.tile<c, 128x128, fp32>
  xsai_ame.store %cs, %single : !xsai_ame.tile<c, 128x128, fp32>, memref<128x128xf32>
  return
}
