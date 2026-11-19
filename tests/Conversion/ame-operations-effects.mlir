// RUN: buddy-opt %s --canonicalize --cse | FileCheck %s --check-prefix=AME
// RUN: buddy-opt %s --lower-ame --canonicalize --cse | FileCheck %s --check-prefix=LLVM --implicit-check-not='"ame.' --implicit-check-not=' = ame.' --implicit-check-not='  ame.'

// Descriptor and arithmetic operations observe AME architectural state. Unused
// results and identical descriptor reads must retain their execution order.
!m = !llvm.target<"riscv.ztt.matrix", i32, 8, 8>

llvm.func @conservative_effects(%request: i64, %matrix: !m, %base: !llvm.ptr) {
  %owner = ame.acquire %request : (i64) -> i64
  %read0 = ame.mgettyp %matrix : (!m) -> i64
  %read1 = ame.mgettyp %matrix : (!m) -> i64
  %configured = ame.msettyp %matrix, %request : (!m, i64) -> !m
  %sum = ame.madd.ew %matrix, %matrix, %matrix : (!m, !m, !m) -> !m
  %loaded = ame.mls.rm %matrix, %base : (!m, !llvm.ptr) -> !m
  ame.release : () -> ()
  llvm.return
}

// AME-LABEL: llvm.func @conservative_effects(
// AME: ame.acquire
// AME: ame.mgettyp
// AME: ame.mgettyp
// AME: ame.msettyp
// AME: ame.madd.ew
// AME: ame.mls.rm
// AME: ame.release
// AME: llvm.return

// LLVM-LABEL: llvm.func @conservative_effects(
// LLVM: llvm.call_intrinsic "llvm.riscv.ztt.ame.acquire"
// LLVM: llvm.call_intrinsic "llvm.riscv.ztt.mgettyp"
// LLVM: llvm.call_intrinsic "llvm.riscv.ztt.mgettyp"
// LLVM: llvm.call_intrinsic "llvm.riscv.ztt.msettyp"
// LLVM: llvm.call_intrinsic "llvm.riscv.ztt.madd.ew"
// LLVM: llvm.call_intrinsic "llvm.riscv.ztt.mls.rm"
// LLVM: llvm.call_intrinsic "llvm.riscv.ztt.ame.release"
// LLVM: llvm.return
