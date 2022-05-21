func.func @main() {
  %v0 = arith.constant dense<[[[0, 1, 2, 3],
                              [4, 5, 6, 7],
                              [8, 9, 10, 11]],
                              [[12, 13, 14, 15],
                              [16, 17, 18, 19],
                              [20, 21, 22, 23]]]> : vector<2x3x4xi32>
  %v1 = vector.extract_strided_slice %v0 
  {offsets = [0, 0], sizes = [2, 2], strides = [1, 1]} :
  vector<2x3x4xi32> to vector<2x2x4xi32>
  vector.print %v1 : vector<2x2x4xi32>

  %v2 = vector.extract_strided_slice %v0
  {offsets = [0, 1], sizes = [2, 2], strides = [1, 1]} :
  vector<2x3x4xi32> to vector<2x2x4xi32>
  vector.print %v2 : vector<2x2x4xi32>
  func.return
   
}