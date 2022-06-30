func.func @main() -> i32 {
  // SEW = 32
  %sew = arith.constant 2 : index
  // LMUL = 2
  %lmul = arith.constant 1 : index
  %avl = arith.constant 70 : index
  %vl = rvv.setvl %avl, %sew, %lmul : index
  vector.print %vl : index

  %ret = arith.constant 0 : i32
  return %ret : i32
}
