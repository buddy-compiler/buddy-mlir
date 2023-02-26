func.func @main() -> i32 {
  // vector.extract can get scalar/sub-vector from a vector.

  // vector.extract only support literal as indices, if you need to extract 
  // something from a vector with runtime values as indices, you need to cast
  // your base vector to 1-D vector and use vector.extractelement instead.

  %base = arith.constant dense<[[0, 1, 2], [10, 11, 12], [20, 21, 22]]> 
    : vector<3x3xi32>

  
  // Extract a scalar:
  %c0 = vector.extract %base[1, 1] : vector<3x3xi32>
  vector.print %c0 : i32


  // Extract a sub-vector:
  %w1 = vector.extract %base[1] : vector<3x3xi32>
  vector.print %w1 : vector<3xi32>


  // For edge case, you can "extract" a vector itself.
  // %w2 will be exactly as same as %base
  %w2 = vector.extract %base[] : vector<3x3xi32>
  vector.print %w2 : vector<3x3xi32>

  %ret = arith.constant 0 : i32
  return %ret : i32
}
