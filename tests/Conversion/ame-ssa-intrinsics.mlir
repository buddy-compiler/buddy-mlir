// RUN: buddy-opt %s --lower-ame | FileCheck %s --check-prefix=LOWER --implicit-check-not='"ame.'
// RUN: buddy-translate -buddy-to-llvmir %s | FileCheck %s --check-prefix=LLVM
// RUN: buddy-translate -buddy-to-llvmir %s | buddy-llc -mtriple=riscv64 -mattr=+xame -verify-machineinstrs -o - | FileCheck %s --check-prefix=ASM
// RUN: buddy-opt %s --lower-ame | buddy-translate -buddy-to-llvmir | FileCheck %s --check-prefix=LLVM
// RUN: buddy-opt %s --lower-ame | buddy-translate -buddy-to-llvmir | buddy-llc -mtriple=riscv64 -mattr=+xame -verify-machineinstrs -o - | FileCheck %s --check-prefix=ASM

// The AME bridge operations carry ZTT target-extension types through MLIR and
// are translated to the XAME LLVM intrinsics without fixed register indices.

llvm.func @matrix(%a: !llvm.target<"riscv.ztt.matrix", i8, 8, 8>,
                  %b: !llvm.target<"riscv.ztt.matrix", i8, 8, 8>,
                  %c: !llvm.target<"riscv.ztt.matrix", i8, 8, 8>)
    -> !llvm.target<"riscv.ztt.matrix", i8, 8, 8> {
  %r = "ame.intr.madd.ew"(%a, %b, %c) :
      (!llvm.target<"riscv.ztt.matrix", i8, 8, 8>,
       !llvm.target<"riscv.ztt.matrix", i8, 8, 8>,
       !llvm.target<"riscv.ztt.matrix", i8, 8, 8>)
      -> !llvm.target<"riscv.ztt.matrix", i8, 8, 8>
  llvm.return %r : !llvm.target<"riscv.ztt.matrix", i8, 8, 8>
}

llvm.func @accumulator(
    %acc: !llvm.target<"riscv.ztt.acc", i32, 8, 8>,
    %a: !llvm.target<"riscv.ztt.matrix", i8, 8, 8>,
    %b: !llvm.target<"riscv.ztt.matrix", i8, 8, 8>)
    -> !llvm.target<"riscv.ztt.acc", i32, 8, 8> {
  %r = "ame.intr.mmulacc.2d"(%acc, %a, %b) :
      (!llvm.target<"riscv.ztt.acc", i32, 8, 8>,
       !llvm.target<"riscv.ztt.matrix", i8, 8, 8>,
       !llvm.target<"riscv.ztt.matrix", i8, 8, 8>)
      -> !llvm.target<"riscv.ztt.acc", i32, 8, 8>
  llvm.return %r : !llvm.target<"riscv.ztt.acc", i32, 8, 8>
}

llvm.func @set_types(
    %m: !llvm.target<"riscv.ztt.matrix", i8, 8, 8>,
    %a: !llvm.target<"riscv.ztt.acc", i8, 8, 8>,
    %type: i64) {
  %m2 = "ame.intr.msettyp"(%m, %type) :
      (!llvm.target<"riscv.ztt.matrix", i8, 8, 8>, i64)
      -> !llvm.target<"riscv.ztt.matrix", i8, 8, 8>
  %a2 = "ame.intr.asettyp"(%a, %type) :
      (!llvm.target<"riscv.ztt.acc", i8, 8, 8>, i64)
      -> !llvm.target<"riscv.ztt.acc", i8, 8, 8>
  llvm.return
}

llvm.func @state(%x: i64) -> i64 {
  %r = "ame.intr.ame.acquire"(%x) : (i64) -> i64
  "ame.intr.ame.release"() : () -> ()
  llvm.return %r : i64
}

// LOWER-LABEL: llvm.func @matrix
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.madd.ew"
// LOWER-LABEL: llvm.func @accumulator
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mmulacc.2d"
// LOWER-LABEL: llvm.func @set_types
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.msettyp"
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.asettyp"
// LOWER-LABEL: llvm.func @state
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.ame.acquire"
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.ame.release"
// LLVM-LABEL: define target("riscv.ztt.matrix", i8, 8, 8) @matrix
// LLVM: call target("riscv.ztt.matrix", i8, 8, 8) @llvm.riscv.ztt.madd.ew
// LLVM-LABEL: define target("riscv.ztt.acc", i32, 8, 8) @accumulator
// LLVM: call target("riscv.ztt.acc", i32, 8, 8) @llvm.riscv.ztt.mmulacc.2d
// LLVM-LABEL: define void @set_types
// LLVM: call target("riscv.ztt.matrix", i8, 8, 8) @llvm.riscv.ztt.msettyp
// LLVM: call target("riscv.ztt.acc", i8, 8, 8) @llvm.riscv.ztt.asettyp
// LLVM: call i64 @llvm.riscv.ztt.ame.acquire.i64
// LLVM: call void @llvm.riscv.ztt.ame.release
// ASM-LABEL: matrix:
// ASM: madd.ew
// ASM-LABEL: accumulator:
// ASM: mmulacc.2d
// ASM-LABEL: set_types:
// ASM: msettyp
// ASM: asettyp
// ASM-LABEL: state:
// ASM: ame.acquire
// ASM: ame.release
