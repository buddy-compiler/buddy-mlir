// RUN: buddy-opt %s --lower-ame | FileCheck %s --check-prefix=LOWER --implicit-check-not='"ame.'
// RUN: buddy-translate -buddy-to-llvmir %s | FileCheck %s --check-prefix=LLVM
// RUN: buddy-opt %s --lower-ame | buddy-translate -buddy-to-llvmir | FileCheck %s --check-prefix=LLVM
// RUN: buddy-opt %s --lower-ame | buddy-translate -buddy-to-llvmir | buddy-llc -mtriple=riscv64 -mattr=+xame -verify-machineinstrs -o - | FileCheck %s --check-prefix=ASM

!m = !llvm.target<"riscv.ztt.matrix", i8, 8, 8>
!a = !llvm.target<"riscv.ztt.acc", i32, 8, 8>
!pair = !llvm.struct<(!m, !m)>

// Result overloads and scalar operands remain independent of XLEN in MLIR.
llvm.func @scalar_state(%x: i32) -> i32 {
  %owned = "ame.intr.ame.acquire"(%x) : (i32) -> i32
  "ame.intr.ame.release"() : () -> ()
  llvm.return %owned : i32
}

// Matrix load/store forms include passthrough values, pointers and strides.
llvm.func @memory(%old: !m, %base: !llvm.ptr, %stride: i32) -> !m {
  %raw = "ame.intr.mls.1r"(%old, %base) : (!m, !llvm.ptr) -> !m
  %rows = "ame.intr.mls.st"(%raw, %base, %stride) :
      (!m, !llvm.ptr, i32) -> !m
  "ame.intr.mss.1r"(%raw, %base) : (!m, !llvm.ptr) -> ()
  "ame.intr.mss.tst"(%rows, %base, %stride) :
      (!m, !llvm.ptr, i32) -> ()
  llvm.return %rows : !m
}

// Two LLVM intrinsic results are carried as one LLVM struct in MLIR.
llvm.func @zip_pair(%left: !m, %right: !m) -> !pair {
  %pair = "ame.intr.mcolzip.ew"(%left, %right) : (!m, !m) -> !pair
  llvm.return %pair : !pair
}

// ACC/M transfers have separately overloaded destination and source types.
llvm.func @moves(%matrix: !m, %acc: !a, %index: i32) -> i32 {
  %moved = "ame.intr.mmov.m.a"(%matrix, %acc) : (!m, !a) -> !m
  %acc2 = "ame.intr.mmov.a.m"(%acc, %moved) : (!a, !m) -> !a
  %type = "ame.intr.agettyp"(%acc2) : (!a) -> i32
  %element = "ame.intr.mmove8.x.m"(%moved, %index) : (!m, i32) -> i32
  %sum = llvm.add %type, %element : i32
  llvm.return %sum : i32
}

// LOWER-LABEL: llvm.func @scalar_state
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.ame.acquire"({{.*}}) : (i32) -> i32
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.ame.release"() : () -> ()
// LOWER-LABEL: llvm.func @memory
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mls.1r"
// LOWER-SAME: -> !llvm.target<"riscv.ztt.matrix", i8, 8, 8>
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mls.st"
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mss.1r"
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mss.tst"
// LOWER-LABEL: llvm.func @zip_pair
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mcolzip.ew"
// LOWER-SAME: -> !llvm.struct<(target<"riscv.ztt.matrix", i8, 8, 8>, target<"riscv.ztt.matrix", i8, 8, 8>)>
// LOWER-LABEL: llvm.func @moves
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mmov.m.a"
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mmov.a.m"
// LOWER-SAME: -> !llvm.target<"riscv.ztt.acc", i32, 8, 8>
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.agettyp"
// LOWER: llvm.call_intrinsic "llvm.riscv.ztt.mmove8.x.m"
// LLVM-LABEL: define i32 @scalar_state
// LLVM: call i32 @llvm.riscv.ztt.ame.acquire.i32
// LLVM: call void @llvm.riscv.ztt.ame.release()
// LLVM-LABEL: define target("riscv.ztt.matrix", i8, 8, 8) @memory
// LLVM: call target("riscv.ztt.matrix", i8, 8, 8) @llvm.riscv.ztt.mls.1r
// LLVM: call target("riscv.ztt.matrix", i8, 8, 8) @llvm.riscv.ztt.mls.st
// LLVM: call void @llvm.riscv.ztt.mss.1r
// LLVM: call void @llvm.riscv.ztt.mss.tst
// LLVM-LABEL: define { target("riscv.ztt.matrix", i8, 8, 8), target("riscv.ztt.matrix", i8, 8, 8) } @zip_pair
// LLVM: call { target("riscv.ztt.matrix", i8, 8, 8), target("riscv.ztt.matrix", i8, 8, 8) } @llvm.riscv.ztt.mcolzip.ew
// LLVM-LABEL: define i32 @moves
// LLVM: call target("riscv.ztt.matrix", i8, 8, 8) @llvm.riscv.ztt.mmov.m.a
// LLVM: call target("riscv.ztt.acc", i32, 8, 8) @llvm.riscv.ztt.mmov.a.m
// LLVM: call i32 @llvm.riscv.ztt.agettyp
// LLVM: call i32 @llvm.riscv.ztt.mmove8.x.m
// ASM-LABEL: scalar_state:
// ASM: ame.acquire
// ASM: ame.release
// ASM-LABEL: memory:
// ASM: mls.1r
// ASM: mls.st
// ASM: mss.1r
// ASM: mss.tst
// ASM-LABEL: zip_pair:
// ASM: mcolzip.ew
// ASM-LABEL: moves:
// ASM: mmov.m.a
// ASM: mmov.a.m
// ASM: agettyp
// ASM: mmove8.x.m
