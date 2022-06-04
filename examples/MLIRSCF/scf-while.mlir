func.func @main() {
  %init = arith.constant 0 : i32
  %end = arith.constant 5 : i32

  %c2_i32 = arith.constant 2 : i32
  %res = scf.while (%arg0 = %init) : (i32) -> (i32) {
    // Before region.
    %cond = arith.cmpi slt, %arg0, %end : i32
    scf.condition(%cond) %arg0 : i32
  } do {
  // After region.
  ^bb0(%arg5: i32):
    %1 = arith.addi %arg5, %c2_i32 : i32
    scf.yield %1 : i32
  }
  vector.print %res : i32
  return
}
