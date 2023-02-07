func.func @main() -> i32 {
  // create_mask is the variable version of constant_mask, without bound 
  // check. It accepts a list of bounds for each dimension, to create a 
  // hyper-rectangular region with 1s, and the rest of 0s.

  %cons1 = arith.constant 1 : index
  %cons2 = arith.constant 2 : index
  %cons3 = arith.constant 3 : index

  %mask0 = vector.create_mask %cons2 : vector<2xi1> // equal to (1,1)
  vector.print %mask0 : vector<2xi1>

  %mask1 = vector.create_mask %cons2, %cons3 : vector<4x4xi1> 
  vector.print %mask1 : vector<4x4xi1>

  %mask2 = vector.create_mask %cons2, %cons1, %cons3 : vector<4x4x4xi1>
  vector.print %mask2 : vector<4x4x4xi1>

  // if the length is out of bound, it will just fill all positions with 1s
  // %mask3 == vector.constant_mask [2, 2] : vector<2x2xi1>
  %mask3 = vector.create_mask %cons3, %cons3 : vector<2x2xi1>
  vector.print %mask3 : vector<2x2xi1> 

  %ret = arith.constant 0 : i32
  return %ret : i32
}
