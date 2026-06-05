// RUN: buddy-opt -verify-diagnostics %s

func.func @reduce_add_requires_float(%input: !vir.vec<?xi32>, %acc: i32) -> i32 {
  // expected-error @+1 {{failed to verify that reduce kind matches the input element type family}}
  %0 = vir.reduce %input, %acc {kind = "add"} : !vir.vec<?xi32>, i32 -> i32
  return %0 : i32
}

// -----

func.func @reduce_maxsi_requires_integer(%input: !vir.vec<?xf32>, %acc: f32) -> f32 {
  // expected-error @+1 {{failed to verify that reduce kind matches the input element type family}}
  %0 = vir.reduce %input, %acc {kind = "maxsi"} : !vir.vec<?xf32>, f32 -> f32
  return %0 : f32
}

// -----

func.func @reduce_unknown_kind_rejected(%input: !vir.vec<?xf32>, %acc: f32) -> f32 {
  // expected-error @+1 {{failed to verify that reduce kind matches the input element type family}}
  %0 = vir.reduce %input, %acc {kind = "mul"} : !vir.vec<?xf32>, f32 -> f32
  return %0 : f32
}
