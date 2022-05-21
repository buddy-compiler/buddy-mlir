func.func @main() {
  %a = arith.constant dense<[1, 2]> : vector<2xi32>
  %b = arith.constant dense<[3, 4]> : vector<2xi32>
  %print_out0 = vector.shuffle %a, %b[0, 1, 2, 3] : vector<2xi32>, vector<2xi32>
  vector.print %print_out0 : vector<4xi32>
  %print_out1 = vector.shuffle %a, %b[3, 2, 1, 0] : vector<2xi32>, vector<2xi32>
  vector.print %print_out1 : vector<4xi32>
  %print_out2 = vector.shuffle %a, %b[0, 3] : vector<2xi32>, vector<2xi32>
  vector.print %print_out2 : vector<2xi32>

  %c = arith.constant dense<3> : vector<3x3xi32>
  %d = arith.constant dense<2> : vector<2x3xi32>
  %print_out3 = vector.shuffle %c, %d[0, 1, 2, 3, 4] : vector<3x3xi32>, vector<2x3xi32>
  vector.print %print_out3 : vector<5x3xi32>
  %print_out4 = vector.shuffle %c, %d[4, 3, 2, 1, 0] : vector<3x3xi32>, vector<2x3xi32>
  vector.print %print_out4 : vector<5x3xi32>
  %print_out5 = vector.shuffle %c, %d[0, 2, 4] : vector<3x3xi32>, vector<2x3xi32>
  vector.print %print_out5 : vector<3x3xi32>
  func.return 

}