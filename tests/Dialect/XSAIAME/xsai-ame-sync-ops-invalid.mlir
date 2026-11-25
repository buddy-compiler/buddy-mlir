// RUN: buddy-opt %s -split-input-file -verify-diagnostics

func.func @reset_result_must_be_sync() {
  // expected-error@+1 {{result #0 must be}}
  %sync = "xsai_ame.msyncregreset"() : () -> i32
  return
}

// -----

func.func @release_operand_must_be_sync(%not_sync: i32) {
  // expected-error@+1 {{operand #0 must be}}
  %next = "xsai_ame.mrelease"(%not_sync) : (i32) -> !xsai_ame.sync
  return
}

// -----

func.func @release_result_must_be_sync(%sync: !xsai_ame.sync) {
  // expected-error@+1 {{result #0 must be}}
  %next = "xsai_ame.mrelease"(%sync) : (!xsai_ame.sync) -> i32
  return
}

// -----

func.func @acquire_operand_must_be_sync(%not_sync: i32, %target: index) {
  // expected-error@+1 {{operand #0 must be}}
  "xsai_ame.macquire"(%not_sync, %target) : (i32, index) -> ()
  return
}

// -----

func.func @acquire_target_must_be_index(%sync: !xsai_ame.sync, %target: i64) {
  // expected-error@+1 {{operand #1 must be index}}
  "xsai_ame.macquire"(%sync, %target) : (!xsai_ame.sync, i64) -> ()
  return
}
