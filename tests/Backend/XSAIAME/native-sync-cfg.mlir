// RUN: buddy-opt %s -lower-xsai-ame -convert-scf-to-cf -convert-cf-to-llvm -convert-func-to-llvm -convert-arith-to-llvm -reconcile-unrealized-casts | buddy-translate --buddy-to-llvmir > %t.ll
// RUN: FileCheck %s --check-prefix=IR < %t.ll
// RUN: llvm-as %t.ll -o %t.bc
// RUN: llc -mtriple=riscv64 -mattr=+zmasync -verify-machineinstrs -O2 -filetype=obj %t.bc -o %t.o
// RUN: llvm-objdump --mattr=+zmasync -d %t.o | FileCheck %s --check-prefix=OBJ
//
// The native fast allocator (-O0) cannot spill sync registers across blocks.
// Exercise structured SSA conversion with the supported optimized allocator;
// this does not claim native sync spill or cross-function ABI support.
// IR-LABEL: define void @native_sync_control
// IR: call target("riscv.xsai.zmasync.sync") @llvm.riscv.xsai.zmasync.msyncregreset.
// IR: call target("riscv.xsai.zmasync.sync") @llvm.riscv.xsai.zmasync.mrelease.
// IR: phi target("riscv.xsai.zmasync.sync")
// IR: phi target("riscv.xsai.zmasync.sync")
// IR: call target("riscv.xsai.zmasync.sync") @llvm.riscv.xsai.zmasync.mrelease.
// IR: call void @llvm.riscv.xsai.zmasync.macquire.
// OBJ-LABEL: <native_sync_control>:
// OBJ: msyncregreset sync{{[0-9]+}}
// OBJ: mrelease sync{{[0-9]+}}
// OBJ: mrelease sync{{[0-9]+}}
// OBJ: macquire sync{{[0-9]+}},
func.func @native_sync_control(%cond: i1, %n: index, %target: index) {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %s0 = xsai_ame.msyncregreset : !xsai_ame.sync
  %selected = scf.if %cond -> (!xsai_ame.sync) {
    %s1 = xsai_ame.mrelease %s0 : !xsai_ame.sync
    scf.yield %s1 : !xsai_ame.sync
  } else {
    scf.yield %s0 : !xsai_ame.sync
  }
  %loop = scf.for %i = %zero to %n step %one iter_args(%s = %selected) -> (!xsai_ame.sync) {
    %next = xsai_ame.mrelease %s : !xsai_ame.sync
    scf.yield %next : !xsai_ame.sync
  }
  xsai_ame.macquire %loop, %target : !xsai_ame.sync, index
  return
}
