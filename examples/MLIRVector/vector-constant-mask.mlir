func.func @main() -> i32 {
  %mask1 = vector.constant_mask [5] : vector<10xi1>
  vector.print %mask1 : vector<10xi1>

  %mask2 = vector.constant_mask [2, 2, 2] : vector<3x3x3xi1>
  vector.print %mask2 : vector<3x3x3xi1>

  %ret = arith.constant 0 : i32
  return %ret : i32
}
