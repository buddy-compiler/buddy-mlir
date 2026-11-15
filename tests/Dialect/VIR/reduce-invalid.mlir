// RUN: buddy-opt %s -split-input-file -verify-diagnostics

func.func @int_kind_on_floats(%v: !vir.vec<?xf32>, %acc: f32) -> f32 {
  // expected-error @+1 {{expects integers for kind minsi}}
  %r = vir.reduce %v, %acc {kind = "minsi"} : !vir.vec<?xf32>, f32 -> f32
  return %r : f32
}

// -----

func.func @float_kind_on_integers(%v: !vir.vec<?xi32>, %acc: i32) -> i32 {
  // expected-error @+1 {{expects floats for kind maxnumf}}
  %r = vir.reduce %v, %acc {kind = "maxnumf"} : !vir.vec<?xi32>, i32 -> i32
  return %r : i32
}

// -----

func.func @unknown_kind(%v: !vir.vec<?xi32>, %acc: i32) -> i32 {
  // expected-error @+1 {{unknown kind max}}
  %r = vir.reduce %v, %acc {kind = "max"} : !vir.vec<?xi32>, i32 -> i32
  return %r : i32
}

// -----

func.func @accumulator_type(%v: !vir.vec<?xi32>, %acc: i64) -> i64 {
  // expected-error @+1 {{expects the accumulator and the result of the element type of the input, 'i32'}}
  %r = vir.reduce %v, %acc {kind = "add"} : !vir.vec<?xi32>, i64 -> i64
  return %r : i64
}

// -----

func.func @float_num_kind_on_integers(%v: !vir.vec<?xi32>, %acc: i32) -> i32 {
  // expected-error @+1 {{expects floats for kind maximumnumf}}
  %r = vir.reduce %v, %acc {kind = "maximumnumf"} : !vir.vec<?xi32>, i32 -> i32
  return %r : i32
}
