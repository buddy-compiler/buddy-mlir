// RUN: buddy-opt %s --lower-ame | FileCheck %s --check-prefix=LOWER --implicit-check-not='"ame.'
// RUN: buddy-translate -buddy-to-llvmir %s | FileCheck %s --check-prefix=LLVM
// RUN: buddy-translate -buddy-to-llvmir %s | buddy-llc -mtriple=riscv64 -mattr=+xame,+xame-gem5 -verify-machineinstrs -o - | FileCheck %s --check-prefix=ASM
// RUN: buddy-opt %s --lower-ame | buddy-translate -buddy-to-llvmir | FileCheck %s --check-prefix=LLVM
// RUN: buddy-opt %s --lower-ame | buddy-translate -buddy-to-llvmir | buddy-llc -mtriple=riscv64 -mattr=+xame,+xame-gem5 -verify-machineinstrs -o - | FileCheck %s --check-prefix=ASM

// The gem5 profile uses 4x4 tiles and supports 128-bit elements.
llvm.func @matrix_4x4_i128(
    %a: !llvm.target<"riscv.ztt.matrix", i128, 4, 4>)
    -> !llvm.target<"riscv.ztt.matrix", i128, 4, 4> {
  %r = "ame.intr.mzero.2d.m"(%a) :
      (!llvm.target<"riscv.ztt.matrix", i128, 4, 4>)
      -> !llvm.target<"riscv.ztt.matrix", i128, 4, 4>
  llvm.return %r : !llvm.target<"riscv.ztt.matrix", i128, 4, 4>
}

llvm.func @acc_4x4_i128(
    %acc: !llvm.target<"riscv.ztt.acc", i128, 4, 4>)
    -> !llvm.target<"riscv.ztt.acc", i128, 4, 4> {
  %r = "ame.intr.mzero.2d.acc"(%acc) :
      (!llvm.target<"riscv.ztt.acc", i128, 4, 4>)
      -> !llvm.target<"riscv.ztt.acc", i128, 4, 4>
  llvm.return %r : !llvm.target<"riscv.ztt.acc", i128, 4, 4>
}

// LOWER-LABEL: llvm.func @matrix_4x4_i128
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mzero.2d.m"
// LOWER-SAME: -> !llvm.target<"riscv.ztt.matrix", i128, 4, 4>
// LOWER-LABEL: llvm.func @acc_4x4_i128
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mzero.2d.acc"
// LOWER-SAME: -> !llvm.target<"riscv.ztt.acc", i128, 4, 4>
// LLVM-LABEL: define target("riscv.ztt.matrix", i128, 4, 4) @matrix_4x4_i128
// LLVM: call target("riscv.ztt.matrix", i128, 4, 4) @llvm.riscv.ztt.mzero.2d.m
// LLVM-LABEL: define target("riscv.ztt.acc", i128, 4, 4) @acc_4x4_i128
// LLVM: call target("riscv.ztt.acc", i128, 4, 4) @llvm.riscv.ztt.mzero.2d.acc
// ASM-LABEL: matrix_4x4_i128:
// ASM: mzero.2d.m m0
// ASM-LABEL: acc_4x4_i128:
// ASM: mzero.2d.acc acc0
