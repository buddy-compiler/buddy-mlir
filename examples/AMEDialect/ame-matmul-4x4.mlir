// ZTT v0.6 through the AME dialect: C += A * B for one 4 x 4 square.
// This shape matches the AME_gem5 profile (32 M registers, 4 ACC registers,
// 32-bit unit). The caller owns the AME and supplies initialized passthrough
// values. Runtime descriptors must match the SSA element types and shapes.
// Ordinary AME operations use their AME.td custom assembly format. The func
// wrapper is lowered separately after --lower-ame; see the makefile stages.

!matrix = !llvm.target<"riscv.ztt.matrix", i32, 4, 4>
!accumulator = !llvm.target<"riscv.ztt.acc", i32, 4, 4>

module {
  func.func @ame_matmul(
      %a_seed: !matrix, %b_seed: !matrix, %c_seed: !matrix,
      %acc_seed: !accumulator, %matrix_descriptor: i64,
      %acc_descriptor: i64, %a_ptr: !llvm.ptr, %b_ptr: !llvm.ptr,
      %c_ptr: !llvm.ptr) {
    // Configure the datatype descriptors without inventing their bit encodings.
    %a_config = ame.msettyp %a_seed, %matrix_descriptor :
        (!matrix, i64) -> !matrix
    %b_config = ame.msettyp %b_seed, %matrix_descriptor :
        (!matrix, i64) -> !matrix
    %c_config = ame.msettyp %c_seed, %matrix_descriptor :
        (!matrix, i64) -> !matrix
    %acc_config = ame.asettyp %acc_seed, %acc_descriptor :
        (!accumulator, i64) -> !accumulator

    // Load row-major 4 x 4 matrices, including the initial C accumulation values.
    %a = ame.mls.rm %a_config, %a_ptr :
        (!matrix, !llvm.ptr) -> !matrix
    %b = ame.mls.rm %b_config, %b_ptr :
        (!matrix, !llvm.ptr) -> !matrix
    %c = ame.mls.rm %c_config, %c_ptr :
        (!matrix, !llvm.ptr) -> !matrix

    // Move C to ACC, multiply-accumulate, and move the result back to M.
    %initial = ame.mmov.a.m %acc_config, %c :
        (!accumulator, !matrix) -> !accumulator
    %result = ame.mmulacc.2d %initial, %a, %b :
        (!accumulator, !matrix, !matrix) -> !accumulator
    %output = ame.mmov.m.a %c, %result :
        (!matrix, !accumulator) -> !matrix
    ame.mss.rm %output, %c_ptr : (!matrix, !llvm.ptr) -> ()
    func.return
  }
}
// RUN: buddy-opt %s | buddy-opt | FileCheck %s --check-prefix=PARSE
// RUN: buddy-opt %s --lower-ame | FileCheck %s --check-prefix=LOWER --implicit-check-not='ame.'
// RUN: buddy-opt %s --lower-ame --convert-linalg-to-loops --lower-affine --convert-scf-to-cf --expand-strided-metadata --lower-affine --convert-cf-to-llvm --convert-arith-to-llvm --convert-math-to-llvm --convert-func-to-llvm --finalize-memref-to-llvm --reconcile-unrealized-casts -o %t.mlir
// RUN: FileCheck %s --check-prefix=EXPORT --implicit-check-not='ame.' --implicit-check-not='func.func' < %t.mlir
// RUN: buddy-translate %t.mlir --buddy-to-llvmir -o %t.ll
// RUN: FileCheck %s --check-prefix=LLVM < %t.ll
// RUN: buddy-llc %t.ll -mtriple=riscv64 -mattr=+m,+xame,+xame-gem5 -verify-machineinstrs -o - | FileCheck %s --check-prefix=ASM
// RUN: buddy-llc %t.ll -mtriple=riscv64 -mattr=+m,+xame,+xame-gem5 -verify-machineinstrs -filetype=obj -o %t.o

// PARSE-LABEL: func.func @ame_matmul(
// PARSE: ame.msettyp
// PARSE: ame.mls.rm
// PARSE: ame.mmulacc.2d
// PARSE: ame.mss.rm
// PARSE: return

// LOWER-LABEL: func.func @ame_matmul(
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.msettyp"
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.asettyp"
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mls.rm"
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mmov.a.m"
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mmulacc.2d"
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mmov.m.a"
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mss.rm"
// LOWER: return

// EXPORT-LABEL: llvm.func @ame_matmul(
// EXPORT: llvm.call_intrinsic "llvm.riscv.ztt.mmulacc.2d"
// EXPORT: llvm.return

// LLVM-LABEL: define void @ame_matmul(
// LLVM: @llvm.riscv.ztt.msettyp.
// LLVM: @llvm.riscv.ztt.asettyp.
// LLVM: [[A:%.*]] = call {{.*}} @llvm.riscv.ztt.mls.rm.
// LLVM: [[B:%.*]] = call {{.*}} @llvm.riscv.ztt.mls.rm.
// LLVM: [[C:%.*]] = call {{.*}} @llvm.riscv.ztt.mls.rm.
// LLVM: [[INITIAL:%.*]] = call {{.*}} @llvm.riscv.ztt.mmov.a.m.{{.*}} [[C]])
// LLVM: [[RESULT:%.*]] = call {{.*}} @llvm.riscv.ztt.mmulacc.2d.{{.*}} [[INITIAL]], {{.*}} [[A]], {{.*}} [[B]])
// LLVM: [[OUTPUT:%.*]] = call {{.*}} @llvm.riscv.ztt.mmov.m.a.{{.*}} [[C]], {{.*}} [[RESULT]])
// LLVM: call void @llvm.riscv.ztt.mss.rm.{{.*}} [[OUTPUT]], ptr
// LLVM: ret void

// ASM-LABEL: ame_matmul:
// ASM: msettyp
// ASM: asettyp
// ASM: mls.rm
// ASM: mmov.a.m
// ASM: mmulacc.2d
// ASM: mmov.m.a
// ASM: mss.rm
// ASM: ret
