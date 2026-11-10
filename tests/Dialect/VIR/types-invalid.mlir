// RUN: buddy-opt %s -split-input-file -verify-diagnostics

// Invalid dynamic vector types and scaling factors.

// expected-error @below {{the ratio of a scaling factor is 'm' (multiplier) or 'f' (fraction), not 'x'}}
func.func private @ratio(!vir.vec<?xi8, x2>)

// -----

// expected-error @below {{the ratio of a scaling factor is 'm' (multiplier) or 'f' (fraction), not 'x'}}
func.func private @attr_ratio(!vir.vec<?xi8, #vir.sf<x2>>)

// -----

// expected-error @below {{expected a scaling factor, 'm' or 'f' and its value (e.g. m2), not 'mq'}}
func.func private @attr_value(!vir.vec<?xi8, #vir.sf<mq>>)

// -----

// expected-error @below {{the value of a scaling factor is 2, 4, 8, 16, 32 or 64, not 3}}
func.func private @value(!vir.vec<?xi8, m3>)

// -----

// expected-error @below {{the value of a scaling factor is 2, 4, 8, 16, 32 or 64, not 128}}
func.func private @value_large(!vir.vec<?xi8, #vir.sf<f128>>)

// -----

// expected-error @below {{expected a scaling factor (e.g. m2 or #vir.sf<m2>)}}
func.func private @not_factor(!vir.vec<?xi8, 5>)

// -----

// expected-error @below {{failed to verify 'elementType': integer or index or floating-point}}
func.func private @element(!vir.vec<?xvector<4xf32>>)

// -----

// expected-error @below {{the dims before the last one of a dynamic vector are static and positive, not 0}}
func.func private @zero_dim(!vir.vec<0x?xf32>)
