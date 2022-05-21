func.func @main() {
  %c0 = arith.constant 0 : i32
  %print_out0 = vector.splat %c0 : vector<8x16xi32>
  vector.print %print_out0 : vector<8x16xi32>
  %print_out1 = arith.constant dense<0> : vector<8x16xi32>
  vector.print %print_out1 : vector<8x16xi32>
  func.return
  
}