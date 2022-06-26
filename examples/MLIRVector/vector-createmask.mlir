func.func @main() -> i32 {
  %cons2 = arith.constant 2 : index
  %mask0 = vector.create_mask %cons2 : vector<2xi1> // equal to (1,1)
  vector.print %mask0 : vector<2xi1>

  %cons3 = arith.constant 3 : index
  %mask1 = vector.create_mask %cons2, %cons3 : vector<4x4xi1> 
  vector.print %mask1 : vector<4x4xi1>

  %cons1 = arith.constant 1 : index
  %mask2 = vector.create_mask %cons2, %cons1, %cons3 : vector<4x4x4xi1>
  vector.print %mask2 : vector<4x4x4xi1>

  %ret = arith.constant 0 : i32
  return %ret : i32
}
