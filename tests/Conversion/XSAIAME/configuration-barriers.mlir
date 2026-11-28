// RUN: buddy-opt %s -lower-xsai-ame | FileCheck %s

func.func private @unknown()

// Only an unchanged extent is removed; a changed K reconfigures K alone, and
// returning to the earlier K must configure it again.
// CHECK-LABEL: func.func @ordered_dedup
// CHECK: llvm.call_intrinsic "llvm.riscv.xsai.zames.msettilemi"
// CHECK: llvm.mlir.constant(64 : i64)
// CHECK: llvm.call_intrinsic "llvm.riscv.xsai.zames.msettileki"
// CHECK: llvm.call_intrinsic "llvm.riscv.xsai.zames.mla"
// CHECK-NOT: llvm.riscv.xsai.zames.msettile
// CHECK: llvm.call_intrinsic "llvm.riscv.xsai.zames.mla"
// CHECK-NOT: llvm.riscv.xsai.zames.msettilemi
// CHECK: llvm.mlir.constant(32 : i64)
// CHECK: llvm.call_intrinsic "llvm.riscv.xsai.zames.msettileki"
// CHECK: llvm.call_intrinsic "llvm.riscv.xsai.zames.mla"
// CHECK-NOT: llvm.riscv.xsai.zames.msettilemi
// CHECK: llvm.mlir.constant(64 : i64)
// CHECK: llvm.call_intrinsic "llvm.riscv.xsai.zames.msettileki"
// CHECK: llvm.call_intrinsic "llvm.riscv.xsai.zames.mla"
func.func @ordered_dedup(%view: memref<?x?xi8>, %half: memref<?x?xf16>) {
  %a = xsai_ame.load %view : memref<?x?xi8> -> !xsai_ame.tile<a, 128x64, i8>
  %b = xsai_ame.load %view : memref<?x?xi8> -> !xsai_ame.tile<a, 128x64, i8>
  %c = xsai_ame.load %half : memref<?x?xf16> -> !xsai_ame.tile<a, 128x32, fp16>
  %d = xsai_ame.load %view : memref<?x?xi8> -> !xsai_ame.tile<a, 128x64, i8>
  return
}

// Plain unknown calls may reconfigure hardware even without an XSAIAME op/name.
// CHECK-LABEL: func.func @unknown_call_barrier
// CHECK: llvm.call_intrinsic "llvm.riscv.xsai.zames.msettilemi"
// CHECK: llvm.call_intrinsic "llvm.riscv.xsai.zames.msettileki"
// CHECK: llvm.call_intrinsic "llvm.riscv.xsai.zames.mla"
// CHECK: call @unknown()
// CHECK: llvm.call_intrinsic "llvm.riscv.xsai.zames.msettilemi"
// CHECK: llvm.call_intrinsic "llvm.riscv.xsai.zames.msettileki"
// CHECK: llvm.call_intrinsic "llvm.riscv.xsai.zames.mla"
func.func @unknown_call_barrier(%view: memref<128x64xi8>) {
  %a = xsai_ame.load %view : memref<128x64xi8> -> !xsai_ame.tile<a, 128x64, i8>
  call @unknown() : () -> ()
  %b = xsai_ame.load %view : memref<128x64xi8> -> !xsai_ame.tile<a, 128x64, i8>
  return
}

// CHECK-LABEL: func.func @unknown_intrinsic_barrier
// CHECK: llvm.call_intrinsic "llvm.riscv.xsai.zames.mla"
// CHECK: llvm.call_intrinsic "llvm.riscv.xsai.zames.minit"
// CHECK: llvm.call_intrinsic "llvm.riscv.xsai.zames.msettilemi"
// CHECK: llvm.call_intrinsic "llvm.riscv.xsai.zames.msettileki"
// CHECK: llvm.call_intrinsic "llvm.riscv.xsai.zames.mla"
func.func @unknown_intrinsic_barrier(%view: memref<128x64xi8>) {
  %a = xsai_ame.load %view : memref<128x64xi8> -> !xsai_ame.tile<a, 128x64, i8>
  llvm.call_intrinsic "llvm.riscv.xsai.zames.minit"() : () -> ()
  %b = xsai_ame.load %view : memref<128x64xi8> -> !xsai_ame.tile<a, 128x64, i8>
  return
}

// No setter may be removed using knowledge from another block or region.
// CHECK-LABEL: func.func @region_barrier
// CHECK: llvm.call_intrinsic "llvm.riscv.xsai.zames.mla"
// CHECK: scf.if
// CHECK: llvm.call_intrinsic "llvm.riscv.xsai.zames.msettilemi"
// CHECK: llvm.call_intrinsic "llvm.riscv.xsai.zames.msettileki"
// CHECK: llvm.call_intrinsic "llvm.riscv.xsai.zames.mla"
// CHECK: }
// CHECK: llvm.call_intrinsic "llvm.riscv.xsai.zames.msettilemi"
// CHECK: llvm.call_intrinsic "llvm.riscv.xsai.zames.msettileki"
// CHECK: llvm.call_intrinsic "llvm.riscv.xsai.zames.mla"
func.func @region_barrier(%view: memref<128x64xi8>, %cond: i1) {
  %a = xsai_ame.load %view : memref<128x64xi8> -> !xsai_ame.tile<a, 128x64, i8>
  scf.if %cond {
    %b = xsai_ame.load %view : memref<128x64xi8> -> !xsai_ame.tile<a, 128x64, i8>
  }
  %c = xsai_ame.load %view : memref<128x64xi8> -> !xsai_ame.tile<a, 128x64, i8>
  return
}

// Equal independently materialized constants (one per lowered op) are
// deduplicated by value, not by the rewriter's SSA allocation order.
// CHECK-LABEL: func.func @constant_identity
// CHECK: llvm.call_intrinsic "llvm.riscv.xsai.zames.msettilemi"
// CHECK: llvm.call_intrinsic "llvm.riscv.xsai.zames.msettileki"
// CHECK: llvm.call_intrinsic "llvm.riscv.xsai.zames.mla"
// CHECK-NOT: llvm.riscv.xsai.zames.msettile
// CHECK: llvm.call_intrinsic "llvm.riscv.xsai.zames.msa"
func.func @constant_identity(%view: memref<128x64xi8>) {
  %a = xsai_ame.load %view : memref<128x64xi8> -> !xsai_ame.tile<a, 128x64, i8>
  xsai_ame.store %a, %view : !xsai_ame.tile<a, 128x64, i8>, memref<128x64xi8>
  return
}

// All predecessor paths must configure their own state. Conversion visitation
// order must not let a sibling block suppress a required setter.
// CHECK-LABEL: func.func @block_dominance
// CHECK: cf.cond_br
// CHECK: ^bb1:
// CHECK: llvm.call_intrinsic "llvm.riscv.xsai.zames.msettilemi"
// CHECK: llvm.call_intrinsic "llvm.riscv.xsai.zames.msettileki"
// CHECK: llvm.call_intrinsic "llvm.riscv.xsai.zames.mla"
// CHECK: ^bb2:
// CHECK: llvm.call_intrinsic "llvm.riscv.xsai.zames.msettilemi"
// CHECK: llvm.call_intrinsic "llvm.riscv.xsai.zames.msettileki"
// CHECK: llvm.call_intrinsic "llvm.riscv.xsai.zames.mla"
// CHECK: ^bb3:
// CHECK: llvm.call_intrinsic "llvm.riscv.xsai.zames.msettilemi"
// CHECK: llvm.call_intrinsic "llvm.riscv.xsai.zames.msettileki"
// CHECK: llvm.call_intrinsic "llvm.riscv.xsai.zames.mla"
func.func @block_dominance(%view: memref<128x64xi8>, %cond: i1) {
  cf.cond_br %cond, ^left, ^right
^left:
  %a = xsai_ame.load %view : memref<128x64xi8> -> !xsai_ame.tile<a, 128x64, i8>
  cf.br ^join
^right:
  %b = xsai_ame.load %view : memref<128x64xi8> -> !xsai_ame.tile<a, 128x64, i8>
  cf.br ^join
^join:
  %c = xsai_ame.load %view : memref<128x64xi8> -> !xsai_ame.tile<a, 128x64, i8>
  return
}

llvm.func @external_llvm()

// CHECK-LABEL: func.func @llvm_call_barrier
// CHECK: llvm.call_intrinsic "llvm.riscv.xsai.zames.mla"
// CHECK: llvm.call @external_llvm()
// CHECK: llvm.call_intrinsic "llvm.riscv.xsai.zames.msettilemi"
// CHECK: llvm.call_intrinsic "llvm.riscv.xsai.zames.msettileki"
// CHECK: llvm.call_intrinsic "llvm.riscv.xsai.zames.mla"
func.func @llvm_call_barrier(%view: memref<128x64xi8>) {
  %a = xsai_ame.load %view : memref<128x64xi8> -> !xsai_ame.tile<a, 128x64, i8>
  llvm.call @external_llvm() : () -> ()
  %b = xsai_ame.load %view : memref<128x64xi8> -> !xsai_ame.tile<a, 128x64, i8>
  return
}

// ZMASYNC orders matrix operations but does not change the tile dimensions.
// CHECK-LABEL: func.func @sync_preserves_configuration
// CHECK: llvm.call_intrinsic "llvm.riscv.xsai.zames.msettilemi"
// CHECK: llvm.call_intrinsic "llvm.riscv.xsai.zames.msettileki"
// CHECK: llvm.call_intrinsic "llvm.riscv.xsai.zames.mla"
// CHECK: llvm.call_intrinsic "llvm.riscv.xsai.zmasync.msyncregreset"
// CHECK: llvm.call_intrinsic "llvm.riscv.xsai.zmasync.mrelease"
// CHECK: llvm.call_intrinsic "llvm.riscv.xsai.zmasync.macquire"
// CHECK: llvm.call_intrinsic "llvm.riscv.xsai.zmasync.mfence"
// CHECK-NOT: llvm.riscv.xsai.zames.msettile
// CHECK: llvm.call_intrinsic "llvm.riscv.xsai.zames.msa"
func.func @sync_preserves_configuration(%view: memref<128x64xi8>, %target: index) {
  %tile = xsai_ame.load %view : memref<128x64xi8> -> !xsai_ame.tile<a, 128x64, i8>
  %s = xsai_ame.msyncregreset : !xsai_ame.sync
  %next = xsai_ame.mrelease %s : !xsai_ame.sync
  xsai_ame.macquire %next, %target : !xsai_ame.sync, index
  xsai_ame.mfence
  xsai_ame.store %tile, %view : !xsai_ame.tile<a, 128x64, i8>, memref<128x64xi8>
  return
}
