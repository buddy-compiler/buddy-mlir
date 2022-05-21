func.func @main() {
  %v0 = arith.constant dense<[[[0, 1, 2],[3, 4, 5],[6, 7, 8]],
                              [[9, 10, 11],[12, 13, 14],[15, 16, 17]],
                              [[18, 19, 20],[21, 22, 23],[24, 25, 26]]]> : vector<3x3x3xi32>
  vector.print %v0 : vector<3x3x3xi32>
  %c0 = arith.constant 1 : i32
  %v1 = vector.insert %c0, %v0[0, 0, 0] : i32 into vector<3x3x3xi32>
  vector.print %v1 : vector<3x3x3xi32>

  %v2 = arith.constant dense<[27, 28, 29]> : vector<3xi32>
  %v3 = vector.insert %v2, %v0[0, 1] : vector<3xi32> into vector<3x3x3xi32>
  vector.print %v3 : vector<3x3x3xi32> 

  %v4 = arith.constant dense<[[27, 28, 29],[30, 31, 32],[33, 34, 35]]> : vector<3x3xi32>
  %v5 = vector.insert %v4, %v0[2] : vector<3x3xi32> into vector<3x3x3xi32>
  vector.print %v5 : vector<3x3x3xi32>
  func.return 

}