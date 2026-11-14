// RUN: buddy-opt %s --lower-xt-ame | FileCheck %s
// RUN: buddy-opt %s --lower-xt-ame --convert-arith-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts | buddy-translate -buddy-to-llvmir | buddy-llc -mtriple=riscv64 -mattr=+xtheadame -o - | FileCheck %s --check-prefix=ASM

// Legacy register indices are physical matrix registers. Scalar operands and
// results still use normal LLVM register allocation. Keep all operations and
// their order even when matrix writes have no explicit SSA users.
func.func @fixed_registers(%value: i64, %index: i64) -> i64 {
  xt_ame.th.mcfg %value : i64
  xt_ame.th.mzero 0
  xt_ame.th.mdupb.m.x 1, %value
  xt_ame.th.mmovw.m.x 2, %value, %index
  xt_ame.th.mmov.mm 3, 1
  xt_ame.th.mmov.mv.i 4, 3[2]
  xt_ame.th.mpack.mm 5, 4, 2
  xt_ame.th.mmacc.w.b 0, 2, 1
  %result = xt_ame.th.mmovw.x.m 0, %index : i64 -> i64
  return %result : i64
}

// CHECK-LABEL: func.func @fixed_registers
// CHECK: llvm.inline_asm has_side_effects "th.mcfg $0"
// CHECK: llvm.inline_asm has_side_effects "th.mzero m0", "~{memory},~{m0},~{m1},~{m2},~{m3},~{m4},~{m5},~{m6},~{m7}"
// CHECK: llvm.inline_asm has_side_effects "th.mdupb.m.x m1, $0"
// CHECK: llvm.inline_asm has_side_effects "th.mmovw.m.x m2, $0, $1"
// CHECK: llvm.inline_asm has_side_effects "th.mmov.mm m3, m1"
// CHECK: llvm.inline_asm has_side_effects ".insn 4, 68026667"
// CHECK: llvm.inline_asm has_side_effects "th.mpack.mm m5, m4, m2"
// CHECK: llvm.inline_asm has_side_effects "th.mmacc.w.b m0, m2, m1"
// CHECK: %[[RESULT:.*]] = llvm.inline_asm has_side_effects "th.mmovw.x.m $0, m0, $1", "=r,r,~{memory},~{m0},~{m1},~{m2},~{m3},~{m4},~{m5},~{m6},~{m7}"
// CHECK: return %[[RESULT]] : i64

// ASM-LABEL: fixed_registers:
// ASM: th.mcfg a0
// ASM: th.mzero m0
// ASM: th.mdupb.m.x m1, a0
// ASM: th.mmovw.m.x m2, a0, a1
// ASM: th.mmov.mm m3, m1
// ASM: .insn 0x4, 68026667
// ASM: th.mpack.mm m5, m4, m2
// ASM: th.mmacc.w.b m0, m2, m1
// ASM: th.mmovw.x.m a0, m0, a1
