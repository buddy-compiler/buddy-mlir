// RUN: buddy-opt %s -lower-xsai-ame | FileCheck %s --implicit-check-not='!xsai_ame.'

// Structural cross-function conversion is supported; native calling conventions
// and spills for tile/sync registers are deliberately not claimed by this test.
// CHECK-LABEL: func.func private @external(!llvm.target<"riscv.xsai.zmasync.sync">) -> !llvm.target<"riscv.xsai.zmasync.sync">
func.func private @external(!xsai_ame.sync) -> !xsai_ame.sync

// CHECK-LABEL: func.func @chain
// CHECK: %[[RESET:.*]] = llvm.call_intrinsic "llvm.riscv.xsai.zmasync.msyncregreset"
// CHECK: %[[FIRST:.*]] = llvm.call_intrinsic "llvm.riscv.xsai.zmasync.mrelease"(%[[RESET]])
// CHECK: %[[SECOND:.*]] = llvm.call_intrinsic "llvm.riscv.xsai.zmasync.mrelease"(%[[FIRST]])
// CHECK: %[[TARGET:.*]] = arith.index_cast
// CHECK: llvm.call_intrinsic "llvm.riscv.xsai.zmasync.macquire"(%[[SECOND]], %[[TARGET]])
// CHECK: llvm.call_intrinsic "llvm.riscv.xsai.zmasync.mfence"
func.func @chain(%target: index) {
  %s0 = xsai_ame.msyncregreset : !xsai_ame.sync
  %s1 = xsai_ame.mrelease %s0 : !xsai_ame.sync
  %s2 = xsai_ame.mrelease %s1 : !xsai_ame.sync
  xsai_ame.macquire %s2, %target : !xsai_ame.sync, index
  xsai_ame.mfence
  return
}

// CHECK-LABEL: func.func @calls(
// CHECK-SAME: %[[ARG:[a-zA-Z0-9_]+]]: !llvm.target<"riscv.xsai.zmasync.sync">) -> !llvm.target<"riscv.xsai.zmasync.sync">
// CHECK: %[[RESULT:.*]] = call @external(%[[ARG]]) : (!llvm.target<"riscv.xsai.zmasync.sync">) -> !llvm.target<"riscv.xsai.zmasync.sync">
// CHECK: return %[[RESULT]] : !llvm.target<"riscv.xsai.zmasync.sync">
func.func @calls(%s: !xsai_ame.sync) -> !xsai_ame.sync {
  %r = call @external(%s) : (!xsai_ame.sync) -> !xsai_ame.sync
  return %r : !xsai_ame.sync
}

// Each successor receives its actual SSA token. No hidden physical-ID map.
// CHECK-LABEL: func.func @diamond
// CHECK: cf.cond_br {{.*}}, ^bb1(%{{.*}} : !llvm.target<"riscv.xsai.zmasync.sync">), ^bb2(%{{.*}} : !llvm.target<"riscv.xsai.zmasync.sync">)
// CHECK: ^bb1(%[[LEFT:.*]]: !llvm.target<"riscv.xsai.zmasync.sync">):
// CHECK: %[[RELEASED:.*]] = llvm.call_intrinsic "llvm.riscv.xsai.zmasync.mrelease"(%[[LEFT]])
// CHECK: cf.br ^bb3(%[[RELEASED]] : !llvm.target<"riscv.xsai.zmasync.sync">)
// CHECK: ^bb2(%[[RIGHT:.*]]: !llvm.target<"riscv.xsai.zmasync.sync">):
// CHECK: cf.br ^bb3(%[[RIGHT]] : !llvm.target<"riscv.xsai.zmasync.sync">)
// CHECK: ^bb3(%[[JOIN:.*]]: !llvm.target<"riscv.xsai.zmasync.sync">):
// CHECK: llvm.call_intrinsic "llvm.riscv.xsai.zmasync.macquire"(%[[JOIN]],
func.func @diamond(%cond: i1, %target: index) {
  %s = xsai_ame.msyncregreset : !xsai_ame.sync
  cf.cond_br %cond, ^left(%s : !xsai_ame.sync), ^right(%s : !xsai_ame.sync)
^left(%l: !xsai_ame.sync):
  %released = xsai_ame.mrelease %l : !xsai_ame.sync
  cf.br ^join(%released : !xsai_ame.sync)
^right(%r: !xsai_ame.sync):
  cf.br ^join(%r : !xsai_ame.sync)
^join(%joined: !xsai_ame.sync):
  xsai_ame.macquire %joined, %target : !xsai_ame.sync, index
  return
}

// CHECK-LABEL: func.func @structured
// CHECK: %[[IF:.*]] = scf.if {{.*}} -> (!llvm.target<"riscv.xsai.zmasync.sync">)
// CHECK: scf.yield {{.*}} : !llvm.target<"riscv.xsai.zmasync.sync">
// CHECK: } else {
// CHECK: scf.yield {{.*}} : !llvm.target<"riscv.xsai.zmasync.sync">
// CHECK: %[[LOOP:.*]] = scf.for {{.*}} iter_args(%[[ITER:.*]] = %[[IF]]) -> (!llvm.target<"riscv.xsai.zmasync.sync">)
// CHECK: %[[NEXT:.*]] = llvm.call_intrinsic "llvm.riscv.xsai.zmasync.mrelease"(%[[ITER]])
// CHECK: scf.yield %[[NEXT]] : !llvm.target<"riscv.xsai.zmasync.sync">
// CHECK: llvm.call_intrinsic "llvm.riscv.xsai.zmasync.macquire"(%[[LOOP]],
func.func @structured(%cond: i1, %n: index, %target: index) {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %s = xsai_ame.msyncregreset : !xsai_ame.sync
  %selected = scf.if %cond -> (!xsai_ame.sync) {
    %released = xsai_ame.mrelease %s : !xsai_ame.sync
    scf.yield %released : !xsai_ame.sync
  } else {
    scf.yield %s : !xsai_ame.sync
  }
  %result = scf.for %i = %zero to %n step %one iter_args(%iter = %selected) -> (!xsai_ame.sync) {
    %next = xsai_ame.mrelease %iter : !xsai_ame.sync
    scf.yield %next : !xsai_ame.sync
  }
  xsai_ame.macquire %result, %target : !xsai_ame.sync, index
  return
}

// CHECK-LABEL: func.func @while_sync
// CHECK: scf.while (%[[BEFORE:.*]] = {{.*}}) : (!llvm.target<"riscv.xsai.zmasync.sync">) -> !llvm.target<"riscv.xsai.zmasync.sync">
// CHECK: scf.condition({{.*}}) %[[BEFORE]] : !llvm.target<"riscv.xsai.zmasync.sync">
// CHECK: ^bb0(%[[AFTER:.*]]: !llvm.target<"riscv.xsai.zmasync.sync">):
// CHECK: llvm.call_intrinsic "llvm.riscv.xsai.zmasync.mrelease"(%[[AFTER]])
func.func @while_sync(%s: !xsai_ame.sync, %cond: i1) -> !xsai_ame.sync {
  %result = scf.while (%before = %s) : (!xsai_ame.sync) -> !xsai_ame.sync {
    scf.condition(%cond) %before : !xsai_ame.sync
  } do {
  ^bb0(%after: !xsai_ame.sync):
    %next = xsai_ame.mrelease %after : !xsai_ame.sync
    scf.yield %next : !xsai_ame.sync
  }
  return %result : !xsai_ame.sync
}

// Function values retain a function type while recursively converting carriers.
// CHECK-LABEL: func.func @indirect
// CHECK: %[[FN:.*]] = {{(func\.)?}}constant @external : (!llvm.target<"riscv.xsai.zmasync.sync">) -> !llvm.target<"riscv.xsai.zmasync.sync">
// CHECK: %[[RESULT:.*]] = call_indirect %[[FN]]({{.*}}) : (!llvm.target<"riscv.xsai.zmasync.sync">) -> !llvm.target<"riscv.xsai.zmasync.sync">
// CHECK: return %[[RESULT]] : !llvm.target<"riscv.xsai.zmasync.sync">
func.func @indirect(%s: !xsai_ame.sync) -> !xsai_ame.sync {
  %fn = func.constant @external : (!xsai_ame.sync) -> !xsai_ame.sync
  %r = func.call_indirect %fn(%s) : (!xsai_ame.sync) -> !xsai_ame.sync
  return %r : !xsai_ame.sync
}

// cf.switch is a branch carrier too, not only cf.br and cf.cond_br.
// CHECK-LABEL: func.func @switch_sync
// CHECK: cf.switch {{.*}} : i32, [
// CHECK: default: ^bb1(%{{.*}} : !llvm.target<"riscv.xsai.zmasync.sync">)
// CHECK: 0: ^bb1(%{{.*}} : !llvm.target<"riscv.xsai.zmasync.sync">)
// CHECK: ^bb1(%[[TOKEN:.*]]: !llvm.target<"riscv.xsai.zmasync.sync">):
// CHECK: return %[[TOKEN]] : !llvm.target<"riscv.xsai.zmasync.sync">
func.func @switch_sync(%s: !xsai_ame.sync, %selector: i32) -> !xsai_ame.sync {
  cf.switch %selector : i32, [
    default: ^join(%s : !xsai_ame.sync),
    0: ^join(%s : !xsai_ame.sync)
  ]
^join(%result: !xsai_ame.sync):
  return %result : !xsai_ame.sync
}

// CHECK-LABEL: func.func @execute_and_select
// CHECK: %[[REGION:.*]] = scf.execute_region -> !llvm.target<"riscv.xsai.zmasync.sync">
// CHECK: cf.br ^bb1(%{{.*}} : !llvm.target<"riscv.xsai.zmasync.sync">)
// CHECK: ^bb1(%[[CARRIED:.*]]: !llvm.target<"riscv.xsai.zmasync.sync">):
// CHECK: scf.yield %[[CARRIED]] : !llvm.target<"riscv.xsai.zmasync.sync">
// CHECK: scf.index_switch {{.*}} -> !llvm.target<"riscv.xsai.zmasync.sync">
// CHECK: case 0
// CHECK: llvm.call_intrinsic "llvm.riscv.xsai.zmasync.mrelease"(%[[REGION]])
// CHECK: default
// CHECK: scf.yield %[[REGION]] : !llvm.target<"riscv.xsai.zmasync.sync">
func.func @execute_and_select(%s: !xsai_ame.sync, %selector: index) -> !xsai_ame.sync {
  %region = scf.execute_region -> !xsai_ame.sync {
    cf.br ^join(%s : !xsai_ame.sync)
  ^join(%carried: !xsai_ame.sync):
    scf.yield %carried : !xsai_ame.sync
  }
  %result = scf.index_switch %selector -> !xsai_ame.sync
  case 0 {
    %next = xsai_ame.mrelease %region : !xsai_ame.sync
    scf.yield %next : !xsai_ame.sync
  }
  default {
    scf.yield %region : !xsai_ame.sync
  }
  return %result : !xsai_ame.sync
}

// Region arguments require conversion even when there is no operation result
// to make the parent SCF operation illegal.
// CHECK-LABEL: func.func @void_execute
// CHECK: scf.execute_region {
// CHECK: cf.br ^bb1(%{{.*}} : !llvm.target<"riscv.xsai.zmasync.sync">)
// CHECK: ^bb1(%[[TOKEN:.*]]: !llvm.target<"riscv.xsai.zmasync.sync">):
// CHECK: llvm.call_intrinsic "llvm.riscv.xsai.zmasync.macquire"(%[[TOKEN]],
func.func @void_execute(%s: !xsai_ame.sync, %target: index) {
  scf.execute_region {
    cf.br ^join(%s : !xsai_ame.sync)
  ^join(%carried: !xsai_ame.sync):
    xsai_ame.macquire %carried, %target : !xsai_ame.sync, index
    scf.yield
  }
  return
}

// Neither the function signature nor execute_region results mention sync.
// Nested CFG block arguments still require full structural conversion.
// CHECK-LABEL: func.func @local_void_execute
// CHECK: %[[RESET:[a-zA-Z0-9_]+]] = llvm.call_intrinsic "llvm.riscv.xsai.zmasync.msyncregreset"
// CHECK: scf.execute_region {
// CHECK: cf.br ^bb1(%[[RESET]] : !llvm.target<"riscv.xsai.zmasync.sync">)
// CHECK: ^bb1(%[[LOCAL:[a-zA-Z0-9_]+]]: !llvm.target<"riscv.xsai.zmasync.sync">):
// CHECK: llvm.call_intrinsic "llvm.riscv.xsai.zmasync.macquire"(%[[LOCAL]],
func.func @local_void_execute(%target: index) {
  %s = xsai_ame.msyncregreset : !xsai_ame.sync
  scf.execute_region {
    cf.br ^join(%s : !xsai_ame.sync)
  ^join(%carried: !xsai_ame.sync):
    xsai_ame.macquire %carried, %target : !xsai_ame.sync, index
    scf.yield
  }
  return
}
