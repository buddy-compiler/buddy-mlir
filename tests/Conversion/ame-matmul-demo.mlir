// RUN: buddy-opt %S/../../examples/AMEDialect/ame-matmul-demo.mlir --lower-ame --convert-scf-to-cf --convert-cf-to-llvm --convert-arith-to-llvm --convert-func-to-llvm --finalize-memref-to-llvm --reconcile-unrealized-casts -o %t.mlir
// RUN: buddy-translate %t.mlir --buddy-to-llvmir -o %t.ll
// RUN: FileCheck %s --check-prefix=LLVM --implicit-check-not='@malloc' --implicit-check-not='@free' < %t.ll
// RUN: buddy-llc %t.ll -mtriple=riscv64 -mattr=+m,+xame,+xame-gem5 -target-abi=lp64 -verify-machineinstrs -o - | FileCheck %s --check-prefix=ASM
// RUN: buddy-llc %t.ll -mtriple=riscv64 -mattr=+m,+xame,+xame-gem5 -target-abi=lp64 -verify-machineinstrs -filetype=obj -o %t.o

// main constructs the initial AME values locally and releases ownership before
// printing; no matrix ABI adapter or heap allocation is needed for this demo.
// LLVM-LABEL: define i32 @main()
// LLVM: alloca
// LLVM: store i32
// LLVM: @llvm.riscv.ztt.ame.acquire
// LLVM: @llvm.riscv.ztt.msettyp.
// LLVM: @llvm.riscv.ztt.asettyp.
// LLVM: @llvm.riscv.ztt.mzero.2d.acc.
// LLVM: @llvm.riscv.ztt.mls.rm.
// LLVM: @llvm.riscv.ztt.mmulacc.2d.
// LLVM: @llvm.riscv.ztt.mss.rm.
// LLVM: @llvm.riscv.ztt.ame.release
// LLVM: call i32 @print_C(
// LLVM: mul i32
// LLVM: icmp ne i32

// ASM-LABEL: main:
// ASM: ame.acquire
// ASM: msettyp
// ASM: asettyp
// ASM: mzero.2d.acc
// ASM: mls.rm
// ASM: mmulacc.2d
// ASM: mss.rm
// ASM: ame.release
// ASM: call print_C
