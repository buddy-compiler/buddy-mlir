func.func @rvv_setvl(%avl: index) -> index {
  %sew = arith.constant 32 : index
  %lmul = arith.constant 2 : index
  %vl = rvv.setvl %avl, %sew, %lmul : index
  return %vl : index
}
