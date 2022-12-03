func.func @main() -> i32 {
  // constant_mask is the constant version of create_mask, with additional bound 
  // check. It accept a list of bound for each dimension, to create a 
  // hyper-rectangular region with 1s, and rest of 0s.

  // This will create a [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  %mask1 = vector.constant_mask [5] : vector<10xi1>
  vector.print %mask1 : vector<10xi1>

  // This will create a 2x2x2 region of 1s, and rest for 0s.
  %mask2 = vector.constant_mask [2, 2, 2] : vector<3x3x3xi1>
  vector.print %mask2 : vector<3x3x3xi1>

  // It will perform bound check, so the IR below is not allowed
  // %mask3 = vector.constant_mask [3, 3] : vector<2x2xi1>

  %ret = arith.constant 0 : i32
  return %ret : i32
}
