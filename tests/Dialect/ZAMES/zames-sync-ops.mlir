// RUN: buddy-opt %s | buddy-opt | FileCheck %s
// RUN: buddy-opt %s -canonicalize | FileCheck %s --check-prefix=CANON

// CHECK-LABEL: func.func @sync_chain
// CANON-LABEL: func.func @sync_chain
func.func @sync_chain(%target: index) {
  // CHECK: %[[S0:.*]] = zames.msyncregreset : !zames.sync
  // CANON: %[[S0:.*]] = zames.msyncregreset : !zames.sync
  %s0 = zames.msyncregreset : !zames.sync
  // CHECK: zames.mfence
  // CANON: zames.mfence
  zames.mfence
  // CHECK: %[[S1:.*]] = zames.mrelease %[[S0]] : !zames.sync
  // CANON: %[[S1:.*]] = zames.mrelease %[[S0]] : !zames.sync
  %s1 = zames.mrelease %s0 : !zames.sync
  // CHECK: %[[S2:.*]] = zames.mrelease %[[S1]] : !zames.sync
  // CANON: %[[S2:.*]] = zames.mrelease %[[S1]] : !zames.sync
  %s2 = zames.mrelease %s1 : !zames.sync
  // CHECK: zames.macquire %[[S2]], %{{.*}} : !zames.sync, index
  // CANON: zames.macquire %[[S2]], %{{.*}} : !zames.sync, index
  zames.macquire %s2, %target : !zames.sync, index
  return
}

// An unused result must not make reset or release dead.
// CHECK-LABEL: func.func @unused_sync_results
// CANON-LABEL: func.func @unused_sync_results
func.func @unused_sync_results(%sync: !zames.sync) {
  // CANON: zames.msyncregreset : !zames.sync
  %unused_reset = zames.msyncregreset : !zames.sync
  // CANON: zames.mrelease %{{.*}} : !zames.sync
  %unused_release = zames.mrelease %sync : !zames.sync
  return
}

// CHECK-LABEL: func.func @sync_successor
func.func @sync_successor(%sync: !zames.sync) -> !zames.sync {
  %next = zames.mrelease %sync : !zames.sync
  return %next : !zames.sync
}

// CHECK-LABEL: func.func @sync_branch
func.func @sync_branch(%condition: i1, %target: index) {
  %sync = zames.msyncregreset : !zames.sync
  cf.cond_br %condition, ^left(%sync : !zames.sync), ^right(%sync : !zames.sync)
^left(%left_sync: !zames.sync):
  %left_next = zames.mrelease %left_sync : !zames.sync
  // CHECK: cf.br ^[[JOINBLOCK:bb[0-9]+]](
  cf.br ^join(%left_next : !zames.sync)
^right(%right_sync: !zames.sync):
  cf.br ^join(%right_sync : !zames.sync)
// CHECK: ^[[JOINBLOCK]](%[[JOIN:.*]]: !zames.sync):
^join(%joined: !zames.sync):
  // CHECK: zames.macquire %[[JOIN]], %{{.*}} : !zames.sync, index
  zames.macquire %joined, %target : !zames.sync, index
  return
}

// CHECK-LABEL: func.func @sync_loop
func.func @sync_loop(%lb: index, %ub: index, %step: index, %target: index) {
  %sync = zames.msyncregreset : !zames.sync
  // CHECK: scf.for {{.*}} iter_args({{.*}}) -> (!zames.sync)
  %last = scf.for %i = %lb to %ub step %step iter_args(%current = %sync) -> (!zames.sync) {
    %next = zames.mrelease %current : !zames.sync
    scf.yield %next : !zames.sync
  }
  zames.macquire %last, %target : !zames.sync, index
  return
}
