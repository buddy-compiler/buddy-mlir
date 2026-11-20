// RUN: buddy-opt %s -split-input-file -verify-diagnostics

func.func @wrong_a_role(%a: !zames.tile<b, 4x16, i8>, %b: !zames.tile<b, 8x16, i8>, %c: !zames.tile<c, 4x8, i32>) {
  // expected-error@+1 {{first operand must have kind a}}
  %r = zames.mma %a, %b, %c : !zames.tile<b, 4x16, i8>, !zames.tile<b, 8x16, i8>, !zames.tile<c, 4x8, i32>
  return
}

// -----

func.func @wrong_b_role(%a: !zames.tile<a, 4x16, i8>, %b: !zames.tile<a, 8x16, i8>, %c: !zames.tile<c, 4x8, i32>) {
  // expected-error@+1 {{second operand must have kind b}}
  %r = zames.mma %a, %b, %c : !zames.tile<a, 4x16, i8>, !zames.tile<a, 8x16, i8>, !zames.tile<c, 4x8, i32>
  return
}

// -----

func.func @wrong_c_role(%a: !zames.tile<a, 4x16, i8>, %b: !zames.tile<b, 8x16, i8>, %c: !zames.tile<a, 4x8, i32>) {
  // expected-error@+1 {{accumulator must have kind c}}
  %r = zames.mma %a, %b, %c : !zames.tile<a, 4x16, i8>, !zames.tile<b, 8x16, i8>, !zames.tile<a, 4x8, i32>
  return
}

// -----

func.func @mismatched_k(%a: !zames.tile<a, 4x16, i8>, %b: !zames.tile<b, 8x32, i8>, %c: !zames.tile<c, 4x8, i32>) {
  // expected-error@+1 {{A.K and B.K static dimensions must match}}
  %r = zames.mma %a, %b, %c : !zames.tile<a, 4x16, i8>, !zames.tile<b, 8x32, i8>, !zames.tile<c, 4x8, i32>
  return
}

// -----

func.func @mismatched_m(%a: !zames.tile<a, 4x16, i8>, %b: !zames.tile<b, 8x16, i8>, %c: !zames.tile<c, 5x8, i32>) {
  // expected-error@+1 {{A.M and C.M static dimensions must match}}
  %r = zames.mma %a, %b, %c : !zames.tile<a, 4x16, i8>, !zames.tile<b, 8x16, i8>, !zames.tile<c, 5x8, i32>
  return
}

// -----

func.func @mismatched_n(%a: !zames.tile<a, 4x16, i8>, %b: !zames.tile<b, 8x16, i8>, %c: !zames.tile<c, 4x9, i32>) {
  // expected-error@+1 {{B.N and C.N static dimensions must match}}
  %r = zames.mma %a, %b, %c : !zames.tile<a, 4x16, i8>, !zames.tile<b, 8x16, i8>, !zames.tile<c, 4x9, i32>
  return
}

// -----

func.func @mma_missing_sizes(%a: !zames.tile<a, ?x?, i8>, %b: !zames.tile<b, ?x?, i8>, %c: !zames.tile<c, ?x?, i32>) {
  // expected-error@+1 {{expected 3 dynamic size operands, but got 0}}
  %r = zames.mma %a, %b, %c : !zames.tile<a, ?x?, i8>, !zames.tile<b, ?x?, i8>, !zames.tile<c, ?x?, i32>
  return
}

// -----

func.func @mma_extra_sizes(%a: !zames.tile<a, 4x16, i8>, %b: !zames.tile<b, 8x16, i8>, %c: !zames.tile<c, 4x8, i32>, %m: i64) {
  // expected-error@+1 {{expected 0 dynamic size operands, but got 1}}
  %r = zames.mma %a, %b, %c[%m] : !zames.tile<a, 4x16, i8>, !zames.tile<b, 8x16, i8>, !zames.tile<c, 4x8, i32>
  return
}

// -----

func.func @zero_missing_size() {
  // expected-error@+1 {{expected 1 dynamic size operands, but got 0}}
  %a = zames.mzero : !zames.tile<a, ?x16, i8>
  return
}

// -----

func.func @zero_extra_size(%m: i64) {
  // expected-error@+1 {{expected 0 dynamic size operands, but got 1}}
  %a = zames.mzero[%m] : !zames.tile<c, 4x8, i32>
  return
}

// -----

func.func @zero_wrong_size_type(%m: index) {
  // expected-note@-1 {{prior use here}}
  // expected-error@+1 {{use of value '%m' expects different type than prior uses: 'i64' vs 'index'}}
  %a = zames.mzero[%m] : !zames.tile<a, ?x16, i8>
  return
}
