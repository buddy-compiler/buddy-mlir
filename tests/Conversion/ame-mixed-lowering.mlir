// RUN: buddy-opt %s --lower-ame | FileCheck %s --check-prefix=LOWER --implicit-check-not='"ame.'
// RUN: buddy-opt %s --lower-ame --convert-scf-to-cf --convert-cf-to-llvm --convert-arith-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts | buddy-translate -buddy-to-llvmir | FileCheck %s --check-prefix=LLVM
// RUN: buddy-opt %s --lower-ame --convert-scf-to-cf --convert-cf-to-llvm --convert-arith-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts | buddy-translate -buddy-to-llvmir | buddy-llc -mtriple=riscv64 -mattr=+xame -verify-machineinstrs -o - | FileCheck %s --check-prefix=ASM

// AME lowering must work inside other dialects without consuming their ops.
// The remaining conversions then carry the matrix SSA value through a branch
// and a function boundary all the way to the XAME backend.
!m = !llvm.target<"riscv.ztt.matrix", i32, 8, 8>

func.func @conditional_matrix(%enabled: i1, %matrix: !m, %x: i64) -> !m {
  %one = arith.constant 1 : i64
  %amount = arith.addi %x, %one : i64
  %result = scf.if %enabled -> (!m) {
    %added = "ame.intr.madd.ew.x"(%matrix, %amount, %matrix) :
        (!m, i64, !m) -> !m
    scf.yield %added : !m
  } else {
    scf.yield %matrix : !m
  }
  return %result : !m
}

// LOWER-LABEL: func.func @conditional_matrix
// LOWER: %[[ONE:.*]] = arith.constant 1 : i64
// LOWER: %[[AMOUNT:.*]] = arith.addi %{{.*}}, %[[ONE]] : i64
// LOWER: %[[RESULT:.*]] = scf.if %{{.*}} -> (!llvm.target<"riscv.ztt.matrix", i32, 8, 8>) {
// LOWER: %[[ADDED:.*]] = llvm.call_intrinsic "llvm.riscv.ztt.madd.ew.x"(%{{.*}}, %[[AMOUNT]], %{{.*}})
// LOWER: scf.yield %[[ADDED]] : !llvm.target<"riscv.ztt.matrix", i32, 8, 8>
// LOWER: } else {
// LOWER: scf.yield
// LOWER: return %[[RESULT]] : !llvm.target<"riscv.ztt.matrix", i32, 8, 8>

// LLVM-LABEL: @conditional_matrix(
// LLVM: add i64
// LLVM: br i1
// LLVM: call target("riscv.ztt.matrix", i32, 8, 8) @llvm.riscv.ztt.madd.ew.x
// LLVM: phi target("riscv.ztt.matrix", i32, 8, 8)
// LLVM: ret target("riscv.ztt.matrix", i32, 8, 8)

// ASM-LABEL: conditional_matrix:
// ASM: madd.ew.x
// ASM: ret
