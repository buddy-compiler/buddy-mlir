// RUN: buddy-opt %s | buddy-opt | FileCheck %s
// RUN: buddy-opt %s -canonicalize | FileCheck %s --check-prefix=CANON

// CHECK-LABEL: func.func @sync_chain
// CANON-LABEL: func.func @sync_chain
func.func @sync_chain(%target: index) {
  // CHECK: %[[S0:.*]] = xsai_ame.msyncregreset : !xsai_ame.sync
  // CANON: %[[S0:.*]] = xsai_ame.msyncregreset : !xsai_ame.sync
  %s0 = xsai_ame.msyncregreset : !xsai_ame.sync
  // CHECK: xsai_ame.mfence
  // CANON: xsai_ame.mfence
  xsai_ame.mfence
  // CHECK: %[[S1:.*]] = xsai_ame.mrelease %[[S0]] : !xsai_ame.sync
  // CANON: %[[S1:.*]] = xsai_ame.mrelease %[[S0]] : !xsai_ame.sync
  %s1 = xsai_ame.mrelease %s0 : !xsai_ame.sync
  // CHECK: %[[S2:.*]] = xsai_ame.mrelease %[[S1]] : !xsai_ame.sync
  // CANON: %[[S2:.*]] = xsai_ame.mrelease %[[S1]] : !xsai_ame.sync
  %s2 = xsai_ame.mrelease %s1 : !xsai_ame.sync
  // CHECK: xsai_ame.macquire %[[S2]], %{{.*}} : !xsai_ame.sync, index
  // CANON: xsai_ame.macquire %[[S2]], %{{.*}} : !xsai_ame.sync, index
  xsai_ame.macquire %s2, %target : !xsai_ame.sync, index
  return
}

// An unused result must not make reset or release dead.
// CHECK-LABEL: func.func @unused_sync_results
// CANON-LABEL: func.func @unused_sync_results
func.func @unused_sync_results(%sync: !xsai_ame.sync) {
  // CANON: xsai_ame.msyncregreset : !xsai_ame.sync
  %unused_reset = xsai_ame.msyncregreset : !xsai_ame.sync
  // CANON: xsai_ame.mrelease %{{.*}} : !xsai_ame.sync
  %unused_release = xsai_ame.mrelease %sync : !xsai_ame.sync
  return
}

// CHECK-LABEL: func.func @sync_successor
func.func @sync_successor(%sync: !xsai_ame.sync) -> !xsai_ame.sync {
  %next = xsai_ame.mrelease %sync : !xsai_ame.sync
  return %next : !xsai_ame.sync
}

// CHECK-LABEL: func.func @sync_branch
func.func @sync_branch(%condition: i1, %target: index) {
  %sync = xsai_ame.msyncregreset : !xsai_ame.sync
  cf.cond_br %condition, ^left(%sync : !xsai_ame.sync), ^right(%sync : !xsai_ame.sync)
^left(%left_sync: !xsai_ame.sync):
  %left_next = xsai_ame.mrelease %left_sync : !xsai_ame.sync
  // CHECK: cf.br ^[[JOINBLOCK:bb[0-9]+]](
  cf.br ^join(%left_next : !xsai_ame.sync)
^right(%right_sync: !xsai_ame.sync):
  cf.br ^join(%right_sync : !xsai_ame.sync)
// CHECK: ^[[JOINBLOCK]](%[[JOIN:.*]]: !xsai_ame.sync):
^join(%joined: !xsai_ame.sync):
  // CHECK: xsai_ame.macquire %[[JOIN]], %{{.*}} : !xsai_ame.sync, index
  xsai_ame.macquire %joined, %target : !xsai_ame.sync, index
  return
}

// CHECK-LABEL: func.func @sync_loop
func.func @sync_loop(%lb: index, %ub: index, %step: index, %target: index) {
  %sync = xsai_ame.msyncregreset : !xsai_ame.sync
  // CHECK: scf.for {{.*}} iter_args({{.*}}) -> (!xsai_ame.sync)
  %last = scf.for %i = %lb to %ub step %step iter_args(%current = %sync) -> (!xsai_ame.sync) {
    %next = xsai_ame.mrelease %current : !xsai_ame.sync
    scf.yield %next : !xsai_ame.sync
  }
  xsai_ame.macquire %last, %target : !xsai_ame.sync, index
  return
}
