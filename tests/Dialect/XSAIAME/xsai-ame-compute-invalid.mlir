// RUN: buddy-opt %s -split-input-file -verify-diagnostics

func.func @wrong_a_role(%a: !xsai_ame.tile<b, 4x16, i8>, %b: !xsai_ame.tile<b, 8x16, i8>, %c: !xsai_ame.tile<c, 4x8, i32>) {
  // expected-error@+1 {{first operand must have kind a}}
  %r = xsai_ame.mma %a, %b, %c : !xsai_ame.tile<b, 4x16, i8>, !xsai_ame.tile<b, 8x16, i8>, !xsai_ame.tile<c, 4x8, i32>
  return
}

// -----

func.func @wrong_b_role(%a: !xsai_ame.tile<a, 4x16, i8>, %b: !xsai_ame.tile<a, 8x16, i8>, %c: !xsai_ame.tile<c, 4x8, i32>) {
  // expected-error@+1 {{second operand must have kind b}}
  %r = xsai_ame.mma %a, %b, %c : !xsai_ame.tile<a, 4x16, i8>, !xsai_ame.tile<a, 8x16, i8>, !xsai_ame.tile<c, 4x8, i32>
  return
}

// -----

func.func @wrong_c_role(%a: !xsai_ame.tile<a, 4x16, i8>, %b: !xsai_ame.tile<b, 8x16, i8>, %c: !xsai_ame.tile<a, 4x8, i32>) {
  // expected-error@+1 {{accumulator must have kind c}}
  %r = xsai_ame.mma %a, %b, %c : !xsai_ame.tile<a, 4x16, i8>, !xsai_ame.tile<b, 8x16, i8>, !xsai_ame.tile<a, 4x8, i32>
  return
}

// -----

func.func @mismatched_k(%a: !xsai_ame.tile<a, 4x16, i8>, %b: !xsai_ame.tile<b, 8x32, i8>, %c: !xsai_ame.tile<c, 4x8, i32>) {
  // expected-error@+1 {{A.K and B.K static dimensions must match}}
  %r = xsai_ame.mma %a, %b, %c : !xsai_ame.tile<a, 4x16, i8>, !xsai_ame.tile<b, 8x32, i8>, !xsai_ame.tile<c, 4x8, i32>
  return
}

// -----

func.func @mismatched_m(%a: !xsai_ame.tile<a, 4x16, i8>, %b: !xsai_ame.tile<b, 8x16, i8>, %c: !xsai_ame.tile<c, 5x8, i32>) {
  // expected-error@+1 {{A.M and C.M static dimensions must match}}
  %r = xsai_ame.mma %a, %b, %c : !xsai_ame.tile<a, 4x16, i8>, !xsai_ame.tile<b, 8x16, i8>, !xsai_ame.tile<c, 5x8, i32>
  return
}

// -----

func.func @mismatched_n(%a: !xsai_ame.tile<a, 4x16, i8>, %b: !xsai_ame.tile<b, 8x16, i8>, %c: !xsai_ame.tile<c, 4x9, i32>) {
  // expected-error@+1 {{B.N and C.N static dimensions must match}}
  %r = xsai_ame.mma %a, %b, %c : !xsai_ame.tile<a, 4x16, i8>, !xsai_ame.tile<b, 8x16, i8>, !xsai_ame.tile<c, 4x9, i32>
  return
}

// -----

func.func @mma_missing_sizes(%a: !xsai_ame.tile<a, ?x?, i8>, %b: !xsai_ame.tile<b, ?x?, i8>, %c: !xsai_ame.tile<c, ?x?, i32>) {
  // expected-error@+1 {{expected 3 dynamic size operands, but got 0}}
  %r = xsai_ame.mma %a, %b, %c : !xsai_ame.tile<a, ?x?, i8>, !xsai_ame.tile<b, ?x?, i8>, !xsai_ame.tile<c, ?x?, i32>
  return
}

// -----

func.func @mma_extra_sizes(%a: !xsai_ame.tile<a, 4x16, i8>, %b: !xsai_ame.tile<b, 8x16, i8>, %c: !xsai_ame.tile<c, 4x8, i32>, %m: i64) {
  // expected-error@+1 {{expected 0 dynamic size operands, but got 1}}
  %r = xsai_ame.mma %a, %b, %c[%m] : !xsai_ame.tile<a, 4x16, i8>, !xsai_ame.tile<b, 8x16, i8>, !xsai_ame.tile<c, 4x8, i32>
  return
}

// -----

func.func @zero_missing_size() {
  // expected-error@+1 {{expected 1 dynamic size operands, but got 0}}
  %a = xsai_ame.mzero : !xsai_ame.tile<a, ?x16, i8>
  return
}

// -----

func.func @zero_extra_size(%m: i64) {
  // expected-error@+1 {{expected 0 dynamic size operands, but got 1}}
  %a = xsai_ame.mzero[%m] : !xsai_ame.tile<c, 4x8, i32>
  return
}

// -----

func.func @zero_wrong_size_type(%m: index) {
  // expected-note@-1 {{prior use here}}
  // expected-error@+1 {{use of value '%m' expects different type than prior uses: 'i64' vs 'index'}}
  %a = xsai_ame.mzero[%m] : !xsai_ame.tile<a, ?x16, i8>
  return
}
