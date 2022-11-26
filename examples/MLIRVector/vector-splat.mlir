func.func @main() -> i32 {
  // vector.splat fills a value into all elements of vector
  %0 = arith.constant 10.0 : f32

  // Creates a vector of shape 3x2, filling %0 as all elements
  %1 = vector.splat %0 : vector<3x2xf32> 
  vector.print %1 : vector<3x2xf32>

  // Doing the same thing with arith.constant
  %2 = arith.constant dense<10.0> : vector<3x2xf32> 
  vector.print %2 : vector<3x2xf32>

  // vector.splat can accept runtime values
  // while arith.constant only accept constants. 

  %ret = arith.constant 0 : i32
  return %ret : i32
}
