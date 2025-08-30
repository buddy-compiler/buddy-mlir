// RUN: buddy-opt -verify-diagnostics %s | buddy-opt | FileCheck %s

// CHECK-LABEL: func.func @rvv_setvl
func.func @rvv_setvl(%avl: index) -> index {
  %sew = arith.constant 32 : index
  %lmul = arith.constant 2 : index
  // CHECK: rvv.setvl{{.*}} : index
  %vl = rvv.setvl %avl, %sew, %lmul : index
  return %vl : index
}
