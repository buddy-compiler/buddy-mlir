// RUN: buddy-opt -verify-diagnostics %s

func.func @extsi_rejects_float_input(%input: !vir.vec<?xf32>) {
  // expected-error@+1 {{input element type must be integer}}
  %0 = vir.extsi %input : !vir.vec<?xf32> -> !vir.vec<?xi32>
  return
}

func.func @extui_rejects_float_result(%input: !vir.vec<?xi8>) {
  // expected-error@+1 {{result element type must be integer}}
  %0 = vir.extui %input : !vir.vec<?xi8> -> !vir.vec<?xf32>
  return
}

func.func @extsi_rejects_shape_mismatch(%input: !vir.vec<2x?xi8>) {
  // expected-error@+1 {{input and result shapes must match}}
  %0 = vir.extsi %input : !vir.vec<2x?xi8> -> !vir.vec<?xi32>
  return
}

func.func @extui_rejects_scaling_factor_mismatch(%input: !vir.vec<?xi8, m2>) {
  // expected-error@+1 {{input and result scaling factors must match}}
  %0 = vir.extui %input : !vir.vec<?xi8, m2> -> !vir.vec<?xi32>
  return
}

func.func @extsi_rejects_non_widening_result(%input: !vir.vec<?xi32>) {
  // expected-error@+1 {{result integer width must be greater than input integer width}}
  %0 = vir.extsi %input : !vir.vec<?xi32> -> !vir.vec<?xi32>
  return
}
