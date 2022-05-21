func.func @main() {
  %v0 = arith.constant dense<[[1,2,3],[4,5,6]]> : vector<2x3xi32> 
  %print_out0 = vector.transpose %v0, [0, 1] : vector<2x3xi32> to vector<2x3xi32>
  vector.print %print_out0 : vector<2x3xi32>
  %print_out1 = vector.transpose %v0, [1, 0] : vector<2x3xi32> to vector<3x2xi32>
  vector.print %print_out1 : vector<3x2xi32>

  %v1 = arith.constant dense<[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
                             [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]]> : vector<2x3x4xi32>
  %print_out2 = vector.transpose %v1, [0, 1, 2] : vector<2x3x4xi32> to vector<2x3x4xi32>
  vector.print %print_out2 : vector<2x3x4xi32>

  %print_out3 = vector.transpose %v1, [2, 1, 0] : vector<2x3x4xi32> to vector<4x3x2xi32>
  vector.print %print_out3 : vector<4x3x2xi32>
  %print_out4 = vector.transpose %v1, [0, 2, 1] : vector<2x3x4xi32> to vector<2x4x3xi32> 
  vector.print %print_out4 : vector<2x4x3xi32> 
  func.return 

}