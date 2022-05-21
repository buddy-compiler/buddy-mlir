func.func @main() {
  %mask0 = vector.constant_mask [2] : vector<5xi1> 
  vector.print %mask0 : vector<5xi1>

  %mask1 = vector.constant_mask [2, 2] : vector<4x4xi1> 
  vector.print %mask1 : vector<4x4xi1>
  func.return
   
}