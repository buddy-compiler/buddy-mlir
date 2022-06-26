func.func @main() -> i32 {
  %0 = arith.constant dense<[[[121, 123, 432, 121], [100, 89, 98, 45], [21, 24, 52, 67], [31, 81, 99, 65]],
                               [[100, 90, 87, 67], [101, 23, 49, 12], [22, 34, 45, 89], [33, 44, 67, 89]],
                               [[100, 90, 97, 67], [112, 113, 45, 95], [122, 78, 67, 54], [33, 56, 89, 64]],
                               [[12, 13, 45, 67], [123, 45, 62, 17], [78, 90, 43, 12], [12, 90, 88, 23]]]> : vector<4x4x4xi32>
    
  %1 = arith.constant dense<[[12, 56, 76],
                             [12, 65, 90],
                             [12, 54, 55]]> : vector<3x3xi32>

  %2 = vector.insert_strided_slice %1, %0 {offsets = [1, 0, 0], strides = [1, 1]} : 
  vector<3x3xi32> into vector<4x4x4xi32>

  vector.print %2 : vector<4x4x4xi32>
  %3 = arith.constant dense<[[23, 56],
                             [451, 651]]> : vector<2x2xi32>

  %4 = vector.insert_strided_slice %3, %1 {offsets = [0, 0], strides = [1, 1]} : 
  vector<2x2xi32> into vector<3x3xi32>
  vector.print %4 : vector<3x3xi32>

  %ret = arith.constant 0 : i32
  return %ret : i32
}
