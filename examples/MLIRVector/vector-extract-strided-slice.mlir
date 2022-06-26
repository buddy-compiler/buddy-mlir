func.func @main() -> i32 {
  %0 = arith.constant dense<[[[1, 2, 3, 4], [1, 7, 8, 9],
                              [12, 11, 14, 19], [100, 101, 112, 111]],
                             [[100, 101, 109, 117], [111, 156, 167, 189],
                              [12, 167, 189, 119], [123, 56, 77, 88]]]> : vector<2x4x4xi32>

  %1 = vector.extract_strided_slice %0 
         {offsets = [0, 0], sizes = [2, 2], strides = [1, 1]} : 
         vector<2x4x4xi32> to vector<2x2x4xi32>

  vector.print %1 : vector<2x2x4xi32>

  %2 = vector.extract_strided_slice %0 
       {offsets = [0, 1], sizes = [2, 2], strides = [1, 1]} : 
       vector<2x4x4xi32> to vector<2x2x4xi32>

  vector.print %2 : vector<2x2x4xi32>

  %ret = arith.constant 0 : i32
  return %ret : i32
}
