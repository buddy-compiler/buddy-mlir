// RUN: buddy-opt %s | FileCheck %s
// RUN: buddy-opt %s | buddy-opt | FileCheck %s

// Dynamic vector types print their scaling factor as the syntax has it, m2,
// which parses back; #vir.sf<m2> is also accepted.

// CHECK-LABEL: func.func private @types(
// CHECK-SAME: !vir.vec<?xf32>, !vir.vec<2x?xi32>, !vir.vec<?xi8, m4>, !vir.vec<?xi64, f2>, !vir.vec<4x?xf16, m64>)
func.func private @types(!vir.vec<?xf32>, !vir.vec<2x?xi32>, !vir.vec<?xi8, m4>,
                         !vir.vec<?xi64, #vir.sf<f2>>, !vir.vec<4x?xf16, m64>)
