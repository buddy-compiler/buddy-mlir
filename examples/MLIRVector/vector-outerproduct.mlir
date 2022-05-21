func.func @main() {
  %v0 = arith.constant dense<[0, 1, 2, 3]> : vector<4xi32>
  %v1 = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7]> : vector<8xi32>
  %print_out0 = vector.outerproduct %v0, %v1 : vector<4xi32>, vector<8xi32>
  vector.print %print_out0 : vector<4x8xi32>

  %c2 = arith.constant 2 : i32
  %print_out1 = vector.outerproduct %v0, %c2 : vector<4xi32>, i32 
  vector.print %print_out1 : vector<4xi32>
  func.return 

}