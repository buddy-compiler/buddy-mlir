func.func @main() {
    
  %cons = arith.constant 2 : index
  %mask0 = vector.create_mask %cons : vector<2xi1> // equal to (1,1)
  vector.print %mask0 : vector<2xi1>

  %cons1 = arith.constant 3 : index
  %mask1 = vector.create_mask %cons, %cons1 : vector<4x4xi1> 
  vector.print %mask1 : vector<4x4xi1>

  %cons2 = arith.constant 3 : index
  %mask2 = vector.create_mask %cons, %cons1, %cons2 : vector<4x4x4xi1>
  vector.print %mask2 : vector<4x4x4xi1>

  return 
}