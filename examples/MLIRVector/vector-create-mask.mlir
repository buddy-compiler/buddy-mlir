func.func @main() {
  %c1 = arith.constant 1 : index 
  %c2 = arith.constant 2 : index 
  %c3 = arith.constant 3 : index 

  %mask0 = vector.create_mask %c1 : vector<3xi1>
  vector.print %mask0 : vector<3xi1>

  %mask1 = vector.create_mask %c2, %c3 : vector<4x4xi1>
  vector.print %mask1 : vector<4x4xi1>
  func.return
   
}