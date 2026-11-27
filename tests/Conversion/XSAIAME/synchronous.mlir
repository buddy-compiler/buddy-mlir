// RUN: buddy-opt %s -lower-xsai-ame="synchronous=true" | FileCheck %s
// RUN: buddy-opt %s -lower-xsai-ame | FileCheck %s --check-prefix=DEFAULT --implicit-check-not=msyncregreset --implicit-check-not=macquire --implicit-check-not=mfence
// RUN: buddy-opt %s -lower-xsai-ame="synchronous=true" -expand-strided-metadata -convert-scf-to-cf -convert-cf-to-llvm -convert-arith-to-llvm -finalize-memref-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts | buddy-translate --buddy-to-llvmir | opt -O3 -S | FileCheck %s --check-prefix=LLVM

// A scalar producer must be visible before a matrix read; a scalar consumer
// must wait for the matching store. Memory clobbers survive LLVM optimization.
// CHECK-LABEL: func.func @serial
// CHECK: memref.store
// CHECK: llvm.inline_asm has_side_effects {{.*}}~{memory}
// CHECK: llvm.call_intrinsic "llvm.riscv.xsai.zmasync.mfence"
// CHECK: llvm.inline_asm has_side_effects {{.*}}~{memory}
// CHECK: llvm.call_intrinsic "llvm.riscv.xsai.zames.mlc"
// CHECK: %[[SYNC:.*]] = llvm.call_intrinsic "llvm.riscv.xsai.zmasync.msyncregreset"
// CHECK: llvm.call_intrinsic "llvm.riscv.xsai.zames.msc"
// CHECK: llvm.inline_asm has_side_effects {{.*}}~{memory}
// CHECK: %[[NEXT:.*]] = llvm.call_intrinsic "llvm.riscv.xsai.zmasync.mrelease"(%[[SYNC]])
// CHECK: llvm.call_intrinsic "llvm.riscv.xsai.zmasync.macquire"(%[[NEXT]],
// CHECK: llvm.inline_asm has_side_effects {{.*}}~{memory}
// CHECK: memref.load
// DEFAULT-LABEL: func.func @serial
// DEFAULT: llvm.call_intrinsic "llvm.riscv.xsai.zames.msc"
// LLVM-LABEL: define float @serial
// LLVM: store float
// LLVM: asm sideeffect "", "~{memory}"
// LLVM: call void @llvm.riscv.xsai.zmasync.mfence
// LLVM: call {{.*}} @llvm.riscv.xsai.zames.mlc
// LLVM: call {{.*}} @llvm.riscv.xsai.zmasync.msyncregreset
// LLVM: call void @llvm.riscv.xsai.zames.msc
// LLVM: call {{.*}} @llvm.riscv.xsai.zmasync.mrelease
// LLVM: call void @llvm.riscv.xsai.zmasync.macquire
// LLVM: asm sideeffect "", "~{memory}"
// LLVM: load float
func.func @serial(%buf: memref<128x128xf32>, %x: f32) -> f32 {
  %zero = arith.constant 0 : index
  memref.store %x, %buf[%zero, %zero] : memref<128x128xf32>
  %tile = xsai_ame.load %buf : memref<128x128xf32> -> !xsai_ame.tile<c, 128x128, fp32>
  xsai_ame.store %tile, %buf : !xsai_ame.tile<c, 128x128, fp32>, memref<128x128xf32>
  %out = memref.load %buf[%zero, %zero] : memref<128x128xf32>
  return %out : f32
}

// Nested matrix stores get a distinct SSA completion chain on every iteration.
// CHECK-LABEL: func.func @nested
// CHECK: scf.for
// CHECK: llvm.call_intrinsic "llvm.riscv.xsai.zmasync.msyncregreset"
// CHECK: llvm.call_intrinsic "llvm.riscv.xsai.zames.msc"
// CHECK: llvm.call_intrinsic "llvm.riscv.xsai.zmasync.mrelease"
// CHECK: llvm.call_intrinsic "llvm.riscv.xsai.zmasync.macquire"
// CHECK: memref.load
// DEFAULT-LABEL: func.func @nested
func.func @nested(%buf: memref<128x128xf32>, %n: index) -> f32 {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %fzero = arith.constant 0.0 : f32
  %result = scf.for %i = %zero to %n step %one iter_args(%sum = %fzero) -> f32 {
    %tile = xsai_ame.mzero : !xsai_ame.tile<c, 128x128, fp32>
    xsai_ame.store %tile, %buf : !xsai_ame.tile<c, 128x128, fp32>, memref<128x128xf32>
    %out = memref.load %buf[%zero, %zero] : memref<128x128xf32>
    %next = arith.addf %sum, %out : f32
    scf.yield %next : f32
  }
  return %result : f32
}
