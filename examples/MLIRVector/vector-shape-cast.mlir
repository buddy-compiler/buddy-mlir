func.func @main() -> i32 {
  %0 = arith.constant dense<0.0> : vector<2x2x3xf32>
  vector.print %0 : vector<2x2x3xf32>

  %1 = vector.shape_cast %0 : vector<2x2x3xf32> to vector<4x3xf32>
  vector.print %1 : vector<4x3xf32>

  %ret = arith.constant 0 : i32
  return %ret : i32
}
