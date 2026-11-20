// RUN: buddy-opt %s -split-input-file -verify-diagnostics

func.func @reset_result_must_be_sync() {
  // expected-error@+1 {{result #0 must be}}
  %sync = "zames.msyncregreset"() : () -> i32
  return
}

// -----

func.func @release_operand_must_be_sync(%not_sync: i32) {
  // expected-error@+1 {{operand #0 must be}}
  %next = "zames.mrelease"(%not_sync) : (i32) -> !zames.sync
  return
}

// -----

func.func @release_result_must_be_sync(%sync: !zames.sync) {
  // expected-error@+1 {{result #0 must be}}
  %next = "zames.mrelease"(%sync) : (!zames.sync) -> i32
  return
}

// -----

func.func @acquire_operand_must_be_sync(%not_sync: i32, %target: index) {
  // expected-error@+1 {{operand #0 must be}}
  "zames.macquire"(%not_sync, %target) : (i32, index) -> ()
  return
}

// -----

func.func @acquire_target_must_be_index(%sync: !zames.sync, %target: i64) {
  // expected-error@+1 {{operand #1 must be index}}
  "zames.macquire"(%sync, %target) : (!zames.sync, i64) -> ()
  return
}
