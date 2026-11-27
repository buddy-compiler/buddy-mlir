// RUN: buddy-opt %s -lower-xsai-ame | FileCheck %s --implicit-check-not='!xsai_ame.'

// Complete tiles configure M/N/K with immediates: M = N = 128, K = 512 bits.
// Partial and dynamic mma tiles are rejected (see unsupported.mlir).
// CHECK-LABEL: func.func @i8_i32
// CHECK: %[[M:.*]] = llvm.mlir.constant(128 : i64)
// CHECK: llvm.call_intrinsic "llvm.riscv.xsai.zames.msettilemi"(%[[M]])
// CHECK: %[[N:.*]] = llvm.mlir.constant(128 : i64)
// CHECK: llvm.call_intrinsic "llvm.riscv.xsai.zames.msettileni"(%[[N]])
// CHECK: %[[K:.*]] = llvm.mlir.constant(64 : i64)
// CHECK: llvm.call_intrinsic "llvm.riscv.xsai.zames.msettileki"(%[[K]])
// CHECK: llvm.call_intrinsic "llvm.riscv.xsai.zames.mmacc"{{.*}} : (!llvm.target<"riscv.xsai.zames.tile.int8">, !llvm.target<"riscv.xsai.zames.tile.int8">, !llvm.target<"riscv.xsai.zames.tile.int32">) -> !llvm.target<"riscv.xsai.zames.tile.int32">
func.func @i8_i32(%a: !xsai_ame.tile<a, 128x64, i8>, %b: !xsai_ame.tile<b, 128x64, i8>, %c: !xsai_ame.tile<c, 128x128, i32>) -> !xsai_ame.tile<c, 128x128, i32> {
  %r = xsai_ame.mma %a, %b, %c : !xsai_ame.tile<a, 128x64, i8>, !xsai_ame.tile<b, 128x64, i8>, !xsai_ame.tile<c, 128x128, i32>
  return %r : !xsai_ame.tile<c, 128x128, i32>
}

// CHECK-LABEL: func.func @e4_fp32
// CHECK: llvm.mlir.constant(64 : i64)
// CHECK: llvm.call_intrinsic "llvm.riscv.xsai.zames.msettileki"
// CHECK: llvm.call_intrinsic "llvm.riscv.xsai.zames.mmacc"{{.*}} : (!llvm.target<"riscv.xsai.zames.tile.fp8e4">, !llvm.target<"riscv.xsai.zames.tile.fp8e4">, !llvm.target<"riscv.xsai.zames.tile.fp32">) -> !llvm.target<"riscv.xsai.zames.tile.fp32">
func.func @e4_fp32(%a: !xsai_ame.tile<a, 128x64, fp8e4m3>, %b: !xsai_ame.tile<b, 128x64, fp8e4m3>, %c: !xsai_ame.tile<c, 128x128, fp32>) -> !xsai_ame.tile<c, 128x128, fp32> {
  %r = xsai_ame.mma %a, %b, %c : !xsai_ame.tile<a, 128x64, fp8e4m3>, !xsai_ame.tile<b, 128x64, fp8e4m3>, !xsai_ame.tile<c, 128x128, fp32>
  return %r : !xsai_ame.tile<c, 128x128, fp32>
}

// CHECK-LABEL: func.func @e5_fp32
// CHECK: llvm.mlir.constant(64 : i64)
// CHECK: llvm.call_intrinsic "llvm.riscv.xsai.zames.msettileki"
// CHECK: llvm.call_intrinsic "llvm.riscv.xsai.zames.mmacc"{{.*}} : (!llvm.target<"riscv.xsai.zames.tile.fp8e5">, !llvm.target<"riscv.xsai.zames.tile.fp8e5">, !llvm.target<"riscv.xsai.zames.tile.fp32">) -> !llvm.target<"riscv.xsai.zames.tile.fp32">
func.func @e5_fp32(%a: !xsai_ame.tile<a, 128x64, fp8e5m2>, %b: !xsai_ame.tile<b, 128x64, fp8e5m2>, %c: !xsai_ame.tile<c, 128x128, fp32>) -> !xsai_ame.tile<c, 128x128, fp32> {
  %r = xsai_ame.mma %a, %b, %c : !xsai_ame.tile<a, 128x64, fp8e5m2>, !xsai_ame.tile<b, 128x64, fp8e5m2>, !xsai_ame.tile<c, 128x128, fp32>
  return %r : !xsai_ame.tile<c, 128x128, fp32>
}

// Half-width elements use K = 32.
// CHECK-LABEL: func.func @half_fp32
// CHECK: llvm.mlir.constant(32 : i64)
// CHECK: llvm.call_intrinsic "llvm.riscv.xsai.zames.msettileki"
// CHECK: llvm.call_intrinsic "llvm.riscv.xsai.zames.mmacc"{{.*}} : (!llvm.target<"riscv.xsai.zames.tile.fp16">, !llvm.target<"riscv.xsai.zames.tile.fp16">, !llvm.target<"riscv.xsai.zames.tile.fp32">) -> !llvm.target<"riscv.xsai.zames.tile.fp32">
func.func @half_fp32(%a: !xsai_ame.tile<a, 128x32, fp16>, %b: !xsai_ame.tile<b, 128x32, fp16>, %c: !xsai_ame.tile<c, 128x128, fp32>) -> !xsai_ame.tile<c, 128x128, fp32> {
  %r = xsai_ame.mma %a, %b, %c : !xsai_ame.tile<a, 128x32, fp16>, !xsai_ame.tile<b, 128x32, fp16>, !xsai_ame.tile<c, 128x128, fp32>
  return %r : !xsai_ame.tile<c, 128x128, fp32>
}

// CHECK-LABEL: func.func @brain_fp32
// CHECK: llvm.mlir.constant(32 : i64)
// CHECK: llvm.call_intrinsic "llvm.riscv.xsai.zames.msettileki"
// CHECK: llvm.call_intrinsic "llvm.riscv.xsai.zames.mmacc"{{.*}} : (!llvm.target<"riscv.xsai.zames.tile.bf16">, !llvm.target<"riscv.xsai.zames.tile.bf16">, !llvm.target<"riscv.xsai.zames.tile.fp32">) -> !llvm.target<"riscv.xsai.zames.tile.fp32">
func.func @brain_fp32(%a: !xsai_ame.tile<a, 128x32, bf16>, %b: !xsai_ame.tile<b, 128x32, bf16>, %c: !xsai_ame.tile<c, 128x128, fp32>) -> !xsai_ame.tile<c, 128x128, fp32> {
  %r = xsai_ame.mma %a, %b, %c : !xsai_ame.tile<a, 128x32, bf16>, !xsai_ame.tile<b, 128x32, bf16>, !xsai_ame.tile<c, 128x128, fp32>
  return %r : !xsai_ame.tile<c, 128x128, fp32>
}
