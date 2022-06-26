func.func @main() -> i32 {
  %0 = arith.constant 10.0 : f32
  %1 = vector.splat %0 : vector<3x2xf32> // creates a vector of shape 3x2, by brodcasting to the respective shape

  vector.print %1 : vector<3x2xf32>
  %2 = arith.constant dense<10.0> : vector<3x2xf32> // Create a vector of shape 3x2, with value 10.0 to test

  vector.print %2 : vector<3x2xf32>

  %ret = arith.constant 0 : i32
  return %ret : i32
}
