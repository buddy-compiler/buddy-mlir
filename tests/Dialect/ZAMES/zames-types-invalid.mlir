// RUN: buddy-opt %s -split-input-file -verify-diagnostics

// expected-error@+1 {{expected tile kind to be one of a, b, or c}}
func.func private @bad_kind(!zames.tile<d, 4x8, i8>)

// -----

// expected-error@+1 {{expected tile dimensions to be positive or dynamic}}
func.func private @zero_first(!zames.tile<a, 0x8, i8>)

// -----

// expected-error@+1 {{expected tile dimensions to be positive or dynamic}}
func.func private @zero_second(!zames.tile<a, 4x0, i8>)

// -----

// expected-error@+1 {{expected ','}}
func.func private @negative_first(!zames.tile<a, -1x8, i8>)

// -----

// expected-error@+1 {{invalid dimension}}
func.func private @negative_second(!zames.tile<a, 4x-2, i8>)

// -----

// Unknown elements must fail even if no ZAMES operation uses the type.
// expected-error@+1 {{unsupported ZAMES element token: f64}}
func.func private @bad_element(!zames.tile<c, 4x8, f64>)

// -----

// expected-error@+1 {{expected tile shape to have exactly 2 dims}}
func.func private @missing_dimension(!zames.tile<a, 4, i8>)

// -----

// expected-error@+1 {{expected tile shape to have exactly 2 dims}}
func.func private @extra_dimension(!zames.tile<a, 4x8x16, i8>)
