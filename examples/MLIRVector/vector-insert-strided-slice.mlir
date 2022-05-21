func.func @main() {
  %v0 = arith.constant dense<[[[0, 1, 2],[3, 4, 5],[6, 7, 8]],
                              [[9, 10, 11],[12, 13, 14],[15, 16, 17]],
                              [[18, 19, 20],[21, 22, 23],[24, 25, 26]]]> : vector<3x3x3xi32>
  %v1 = arith.constant dense<[[520, 521],[522, 523]]> : vector<2x2xi32>
  %v2 = vector.insert_strided_slice %v1, %v0 
  {offsets = [0, 1, 0], strides = [1, 1]} : 
  vector<2x2xi32> into vector<3x3x3xi32>
  vector.print %v2 : vector<3x3x3xi32>

  %v3 = arith.constant dense<[[0, 1, 2, 3], [4, 5, 6, 7], 
                              [8, 9, 10, 11], [12, 13, 14, 15]]> : vector<4x4xi32> 
  %v4 = vector.insert_strided_slice %v1, %v3
  {offsets = [0, 0], strides = [1, 1]} :
  vector<2x2xi32> into vector<4x4xi32>
  vector.print %v4 : vector<4x4xi32>
  func.return 
  
}