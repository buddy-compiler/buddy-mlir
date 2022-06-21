// RUN: buddy-opt %s -lower-rvv -convert-func-to-llvm | buddy-opt | FileCheck %s

func.func @rvv_setvl(%avl: index) -> index {
  %sew = arith.constant 32 : index
  %lmul = arith.constant 2 : index
  // CHECK: rvv.intr.vsetvli{{.*}} : (i64, i64, i64) -> i64
  %vl = rvv.setvl %avl, %sew, %lmul : index
  return %vl : index
}
